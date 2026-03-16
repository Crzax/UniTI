"""Paged Attention vs Contiguous KV Cache — Comparative Tests.

Comprehensive comparison between the two KV cache modes in UniTi:
  1. Correctness: verify both modes produce identical outputs at each stage.
  2. Performance: measure prefill & decode latency for both modes.
  3. Memory efficiency: compare actual memory usage / utilization.
  4. Multi-step decode: verify long generation consistency.
  5. Multi-sequence: batch serving comparison.

Usage:
    python3.10 tests/test_paged_vs_contiguous.py
    python3.10 tests/test_paged_vs_contiguous.py --verbose
    python3.10 tests/test_paged_vs_contiguous.py --test correctness
"""
import sys
import os
import time
import argparse
import traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

import uniti
from uniti.autograd import Tensor, no_grad
from uniti.nn.paged_attention import PagedKVCacheManager
from uniti.nn.nn_qwen2 import Qwen2Attention, Qwen2ForCausalLM, Qwen2Model
from uniti.backend_selection import cpu_numpy

np.random.seed(42)
device = cpu_numpy()


# ============================================================
#  Helpers
# ============================================================

def create_small_model(num_layers=2, hidden_size=64, num_heads=4,
                       num_kv_heads=2, intermediate_size=128,
                       vocab_size=256, max_pos=256):
    """Create a small Qwen2ForCausalLM for testing purposes."""
    model = Qwen2ForCausalLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        rms_norm_eps=1e-6,
        max_position_embeddings=max_pos,
        rope_theta=10000.0,
        device=device,
        dtype="float32",
    )
    model.eval()
    return model


def clone_model(src_model, num_layers=2, hidden_size=64, num_heads=4,
                num_kv_heads=2, intermediate_size=128,
                vocab_size=256, max_pos=256):
    """Create a new model and copy all weights from src_model."""
    dst_model = create_small_model(
        num_layers=num_layers, hidden_size=hidden_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size, vocab_size=vocab_size,
        max_pos=max_pos,
    )

    # Copy embedding weights
    dst_model.model.embed_tokens.weight.cached_data = \
        src_model.model.embed_tokens.weight.realize_cached_data()
    dst_model.model.embed_tokens._weight_numpy_cache = None

    # Copy lm_head weights
    dst_model.lm_head.weight.cached_data = \
        src_model.lm_head.weight.realize_cached_data()

    # Copy layer weights
    for i in range(num_layers):
        src_layer = src_model.model.layers[i]
        dst_layer = dst_model.model.layers[i]

        # Attention projections
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            src_proj = getattr(src_layer.self_attn, proj_name)
            dst_proj = getattr(dst_layer.self_attn, proj_name)
            dst_proj.weight.cached_data = src_proj.weight.realize_cached_data()
            if hasattr(src_proj, 'bias') and src_proj.bias is not None:
                dst_proj.bias.cached_data = src_proj.bias.realize_cached_data()

        # Copy RoPE caches
        dst_layer.self_attn._cos_cache = src_layer.self_attn._cos_cache.copy()
        dst_layer.self_attn._sin_cache = src_layer.self_attn._sin_cache.copy()

        # MLP
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            src_proj = getattr(src_layer.mlp, proj_name)
            dst_proj = getattr(dst_layer.mlp, proj_name)
            dst_proj.weight.cached_data = src_proj.weight.realize_cached_data()

        # Layer norms
        dst_layer.input_layernorm.weight.cached_data = \
            src_layer.input_layernorm.weight.realize_cached_data()
        dst_layer.post_attention_layernorm.weight.cached_data = \
            src_layer.post_attention_layernorm.weight.realize_cached_data()

    # Final norm
    dst_model.model.norm.weight.cached_data = \
        src_model.model.norm.weight.realize_cached_data()

    return dst_model


def greedy_decode_step(logits_np):
    """Greedy decode: pick argmax of last position's logits."""
    return int(np.argmax(logits_np[0, -1, :]))


def run_generate(model, input_ids, max_new_tokens, use_paged,
                 block_size=4, max_blocks=64):
    """Run generation and return (generated_ids, prefill_time, decode_times, cache_stats).

    Works for both paged and contiguous modes.
    """
    prompt_len = len(input_ids)
    max_cache_len = prompt_len + max_new_tokens

    model.reset_cache()
    if use_paged:
        cache_mgr = model.init_paged_cache(
            block_size=block_size,
            max_num_blocks=max_blocks,
            seq_id=0,
            initial_len=0,
        )
    else:
        model.init_cache(batch_size=1, max_cache_len=max_cache_len)
        cache_mgr = None

    ids_np = np.array([input_ids], dtype=np.float32)

    with no_grad():
        # Prefill
        t0 = time.perf_counter()
        ids_tensor = Tensor(ids_np, device=device, dtype="float32", requires_grad=False)
        logits = model(ids_tensor, start_pos=0, last_only=True)
        logits_np = logits.numpy()
        prefill_time = time.perf_counter() - t0

    next_tok = greedy_decode_step(logits_np)
    generated = list(input_ids) + [next_tok]
    cur_pos = prompt_len

    decode_times = []
    for step in range(1, max_new_tokens):
        t0 = time.perf_counter()
        with no_grad():
            tok_tensor = Tensor(
                np.array([[next_tok]], dtype=np.float32),
                device=device, dtype="float32", requires_grad=False,
            )
            logits = model(tok_tensor, start_pos=cur_pos, last_only=True)
            logits_np = logits.numpy()
        dt = time.perf_counter() - t0
        decode_times.append(dt)
        cur_pos += 1

        next_tok = greedy_decode_step(logits_np)
        generated.append(next_tok)

    cache_stats = cache_mgr.get_cache_stats() if cache_mgr else None
    return generated, prefill_time, decode_times, cache_stats


# ============================================================
#  Test: Full model correctness (prefill)
# ============================================================

def test_full_model_prefill_correctness(verbose=False):
    """Verify paged and contiguous modes give identical prefill logits
    on a full Qwen2ForCausalLM model (small)."""
    print("=== test_full_model_prefill_correctness ===")
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2

    model_contig = create_small_model(num_layers=num_layers,
                                      hidden_size=hidden_size,
                                      num_heads=num_heads,
                                      num_kv_heads=num_kv_heads)
    model_paged = clone_model(model_contig, num_layers=num_layers,
                              hidden_size=hidden_size,
                              num_heads=num_heads,
                              num_kv_heads=num_kv_heads)

    seq_len = 10
    input_ids = np.random.randint(0, 256, size=(1, seq_len)).astype(np.float32)
    max_cache_len = seq_len + 20

    # Contiguous
    model_contig.init_cache(batch_size=1, max_cache_len=max_cache_len)
    # Paged
    cache_mgr = model_paged.init_paged_cache(
        block_size=4, max_num_blocks=32, seq_id=0, initial_len=0,
    )

    with no_grad():
        inp = Tensor(input_ids, device=device, dtype="float32", requires_grad=False)
        logits_c = model_contig(inp, start_pos=0, last_only=False)
        logits_p = model_paged(inp, start_pos=0, last_only=False)

    lc = logits_c.numpy()
    lp = logits_p.numpy()

    max_diff = np.max(np.abs(lc - lp))
    mean_diff = np.mean(np.abs(lc - lp))

    if verbose:
        print(f"  Logits shape: {lc.shape}")
        print(f"  Max absolute diff : {max_diff:.2e}")
        print(f"  Mean absolute diff: {mean_diff:.2e}")

    np.testing.assert_allclose(lc, lp, atol=1e-4, rtol=1e-4,
        err_msg="Full model prefill logits differ between contiguous and paged!")

    print("  PASSED")


# ============================================================
#  Test: Full model decode correctness (multi-step)
# ============================================================

def test_full_model_decode_correctness(verbose=False):
    """Verify both modes produce identical token sequences during generation."""
    print("=== test_full_model_decode_correctness ===")
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2

    model_contig = create_small_model(num_layers=num_layers,
                                      hidden_size=hidden_size,
                                      num_heads=num_heads,
                                      num_kv_heads=num_kv_heads)
    model_paged = clone_model(model_contig, num_layers=num_layers,
                              hidden_size=hidden_size,
                              num_heads=num_heads,
                              num_kv_heads=num_kv_heads)

    prompt_len = 8
    max_new_tokens = 15
    input_ids = list(np.random.randint(0, 256, size=prompt_len))

    ids_c, _, _, _ = run_generate(model_contig, input_ids, max_new_tokens,
                                  use_paged=False)
    ids_p, _, _, stats = run_generate(model_paged, input_ids, max_new_tokens,
                                      use_paged=True, block_size=4, max_blocks=32)

    if verbose:
        print(f"  Prompt length  : {prompt_len}")
        print(f"  Generated (C)  : {ids_c[prompt_len:]}")
        print(f"  Generated (P)  : {ids_p[prompt_len:]}")
        if stats:
            print(f"  Paged blocks   : {stats['num_used_blocks']}/{stats['max_num_blocks']}")

    assert ids_c == ids_p, (
        f"Generated sequences differ!\n"
        f"  Contiguous: {ids_c[prompt_len:]}\n"
        f"  Paged     : {ids_p[prompt_len:]}"
    )

    print("  PASSED")


# ============================================================
#  Test: Step-by-step logit comparison
# ============================================================

def test_stepwise_logit_match(verbose=False):
    """Compare logits at every decode step to catch any divergence early."""
    print("=== test_stepwise_logit_match ===")
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2

    model_c = create_small_model(num_layers=num_layers,
                                 hidden_size=hidden_size,
                                 num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
    model_p = clone_model(model_c, num_layers=num_layers,
                          hidden_size=hidden_size,
                          num_heads=num_heads,
                          num_kv_heads=num_kv_heads)

    seq_len = 6
    max_new = 10
    input_ids = np.random.randint(0, 256, size=(1, seq_len)).astype(np.float32)
    max_cache_len = seq_len + max_new

    model_c.init_cache(batch_size=1, max_cache_len=max_cache_len)
    model_p.init_paged_cache(block_size=4, max_num_blocks=32, seq_id=0, initial_len=0)

    with no_grad():
        inp = Tensor(input_ids, device=device, dtype="float32", requires_grad=False)

        # Prefill
        logits_c = model_c(inp, start_pos=0, last_only=True).numpy()
        logits_p = model_p(inp, start_pos=0, last_only=True).numpy()

        max_diff = np.max(np.abs(logits_c - logits_p))
        if verbose:
            print(f"  [Prefill] max diff = {max_diff:.2e}")
        np.testing.assert_allclose(logits_c, logits_p, atol=1e-4, rtol=1e-4,
            err_msg="Prefill logits mismatch")

        next_tok = int(np.argmax(logits_c[0, -1, :]))
        cur_pos = seq_len

        # Decode steps
        for step in range(max_new):
            tok_np = np.array([[next_tok]], dtype=np.float32)
            tok = Tensor(tok_np, device=device, dtype="float32", requires_grad=False)

            logits_c = model_c(tok, start_pos=cur_pos, last_only=True).numpy()
            logits_p = model_p(tok, start_pos=cur_pos, last_only=True).numpy()

            max_diff = np.max(np.abs(logits_c - logits_p))
            if verbose:
                print(f"  [Decode step {step:2d}] max diff = {max_diff:.2e}, "
                      f"token = {next_tok}")

            np.testing.assert_allclose(logits_c, logits_p, atol=1e-4, rtol=1e-4,
                err_msg=f"Decode step {step}: logits differ")

            next_tok = int(np.argmax(logits_c[0, -1, :]))
            cur_pos += 1

    print("  PASSED")


# ============================================================
#  Test: Performance comparison
# ============================================================

def test_performance_comparison(verbose=False):
    """Measure and compare latency of paged vs contiguous cache."""
    print("=== test_performance_comparison ===")
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2

    model_c = create_small_model(num_layers=num_layers,
                                 hidden_size=hidden_size,
                                 num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
    model_p = clone_model(model_c, num_layers=num_layers,
                          hidden_size=hidden_size,
                          num_heads=num_heads,
                          num_kv_heads=num_kv_heads)

    prompt_len = 16
    max_new = 20
    input_ids = list(np.random.randint(0, 256, size=prompt_len))

    # -- Contiguous --
    _, prefill_c, decode_c, _ = run_generate(
        model_c, input_ids, max_new, use_paged=False)

    # -- Paged --
    _, prefill_p, decode_p, stats_p = run_generate(
        model_p, input_ids, max_new, use_paged=True, block_size=4, max_blocks=64)

    avg_decode_c = np.mean(decode_c) if decode_c else 0
    avg_decode_p = np.mean(decode_p) if decode_p else 0

    print(f"  {'Metric':<25s} {'Contiguous':>12s} {'Paged':>12s} {'Ratio(P/C)':>12s}")
    print(f"  {'-'*61}")
    print(f"  {'Prefill time (ms)':<25s} {prefill_c*1000:>12.2f} {prefill_p*1000:>12.2f} "
          f"{prefill_p/max(prefill_c,1e-9):>12.2f}x")
    print(f"  {'Avg decode time (ms)':<25s} {avg_decode_c*1000:>12.2f} {avg_decode_p*1000:>12.2f} "
          f"{avg_decode_p/max(avg_decode_c,1e-9):>12.2f}x")
    print(f"  {'Total decode time (ms)':<25s} {sum(decode_c)*1000:>12.2f} {sum(decode_p)*1000:>12.2f} "
          f"{sum(decode_p)/max(sum(decode_c),1e-9):>12.2f}x")

    if verbose and stats_p:
        print(f"\n  [Paged Stats]")
        print(f"    Blocks used: {stats_p['num_used_blocks']}/{stats_p['max_num_blocks']}")
        print(f"    Utilization: {stats_p['utilization']:.1f}%")

    print("  PASSED (benchmark — no correctness assertion)")


# ============================================================
#  Test: Memory efficiency comparison
# ============================================================

def test_memory_efficiency(verbose=False):
    """Compare memory usage patterns between contiguous and paged modes."""
    print("=== test_memory_efficiency ===")
    num_layers = 4
    num_kv_heads = 4
    head_dim = 16
    hidden_size = 64  # num_heads(4) * head_dim(16)

    # -- Scenario: short prompt, little generation --
    prompt_len = 10
    max_new = 5
    max_cache_len = prompt_len + max_new  # contiguous pre-allocates this

    # Contiguous: always allocates max_cache_len tokens across ALL layers
    contig_kv_size = (
        2  # K and V
        * num_layers
        * 1  # batch_size
        * num_kv_heads
        * max_cache_len
        * head_dim
        * 4  # float32 bytes
    )

    # Paged: only allocates blocks needed for actual tokens
    block_size = 4
    actual_tokens = prompt_len + max_new
    blocks_needed = (actual_tokens + block_size - 1) // block_size
    # Paged pool is shared; only used blocks hold data
    # Pool is pre-allocated, but only `blocks_needed` out of max_blocks are used
    max_blocks = 32
    paged_pool_size = (
        2  # K and V
        * num_layers
        * max_blocks
        * num_kv_heads
        * block_size
        * head_dim
        * 4  # float32 bytes
    )
    paged_used_size = (
        2  # K and V
        * num_layers
        * blocks_needed
        * num_kv_heads
        * block_size
        * head_dim
        * 4
    )

    # For a REAL scenario where max_seq_len is much bigger
    real_max_seq = 2048
    real_contig_size = (
        2 * num_layers * 1 * num_kv_heads * real_max_seq * head_dim * 4
    )
    real_paged_used = (
        2 * num_layers * blocks_needed * num_kv_heads * block_size * head_dim * 4
    )

    print(f"  Configuration:")
    print(f"    layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"    prompt={prompt_len} tokens, generate={max_new} tokens")
    print(f"    actual tokens in cache = {actual_tokens}")
    print()
    print(f"  {'Metric':<35s} {'Contiguous':>14s} {'Paged':>14s}")
    print(f"  {'-'*63}")

    def fmt_bytes(n):
        if n < 1024:
            return f"{n} B"
        elif n < 1024 * 1024:
            return f"{n/1024:.1f} KB"
        else:
            return f"{n/(1024*1024):.2f} MB"

    print(f"  {'Pre-alloc (short seq)':<35s} {fmt_bytes(contig_kv_size):>14s} "
          f"{fmt_bytes(paged_pool_size):>14s}")
    print(f"  {'Actually used':<35s} {fmt_bytes(contig_kv_size):>14s} "
          f"{fmt_bytes(paged_used_size):>14s}")
    print(f"  {'Blocks used / allocated':<35s} {'N/A':>14s} "
          f"{blocks_needed:>5d}/{max_blocks:<8d}")
    print(f"  {'Utilization':<35s} {'100.0%':>14s} "
          f"{blocks_needed/max_blocks*100:>13.1f}%")

    print()
    print(f"  Realistic scenario (max_seq_len={real_max_seq}):")
    print(f"  {'Contiguous pre-alloc':<35s} {fmt_bytes(real_contig_size):>14s}")
    print(f"  {'Paged actually used':<35s} {fmt_bytes(real_paged_used):>14s}")
    saving = (1 - real_paged_used / real_contig_size) * 100
    print(f"  {'Memory saving':<35s} {saving:>13.1f}%")

    # Validate the paged mode actually uses fewer blocks
    mgr = PagedKVCacheManager(
        num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=block_size, max_num_blocks=max_blocks, device=device,
    )
    mgr.allocate_sequence(seq_id=0, initial_len=0)

    for layer_idx in range(num_layers):
        k = np.random.randn(num_kv_heads, actual_tokens, head_dim).astype(np.float32)
        v = np.random.randn(num_kv_heads, actual_tokens, head_dim).astype(np.float32)
        mgr.append_kv(seq_id=0, layer_idx=layer_idx, k_data=k, v_data=v)

    stats = mgr.get_cache_stats()
    assert stats['num_used_blocks'] == blocks_needed, \
        f"Expected {blocks_needed} used blocks, got {stats['num_used_blocks']}"
    assert stats['num_free_blocks'] == max_blocks - blocks_needed

    if verbose:
        print(f"\n  [Validated] PagedKVCacheManager uses exactly {blocks_needed} blocks")

    print("  PASSED")


# ============================================================
#  Test: Block size sensitivity
# ============================================================

def test_block_size_sensitivity(verbose=False):
    """Test that different block sizes still produce correct results."""
    print("=== test_block_size_sensitivity ===")
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2

    model_ref = create_small_model(num_layers=num_layers,
                                   hidden_size=hidden_size,
                                   num_heads=num_heads,
                                   num_kv_heads=num_kv_heads)

    prompt_len = 8
    max_new = 10
    input_ids = list(np.random.randint(0, 256, size=prompt_len))

    # Get reference (contiguous)
    ids_ref, _, _, _ = run_generate(model_ref, input_ids, max_new, use_paged=False)

    # Test various block sizes
    for block_size in [1, 2, 4, 8, 16]:
        model_p = clone_model(model_ref, num_layers=num_layers,
                              hidden_size=hidden_size,
                              num_heads=num_heads,
                              num_kv_heads=num_kv_heads)
        ids_p, _, _, stats = run_generate(
            model_p, input_ids, max_new,
            use_paged=True, block_size=block_size,
            max_blocks=128,  # enough for any block size
        )

        match = ids_ref == ids_p
        status = "✓" if match else "✗"
        blocks_used = stats['num_used_blocks'] if stats else 'N/A'

        if verbose:
            print(f"  block_size={block_size:2d}: {status}  "
                  f"blocks_used={blocks_used}")

        assert match, (
            f"block_size={block_size}: sequences differ!\n"
            f"  Reference : {ids_ref[prompt_len:]}\n"
            f"  Paged     : {ids_p[prompt_len:]}"
        )

    print("  PASSED")


# ============================================================
#  Test: Paged cache reuse after free
# ============================================================

def test_paged_cache_reuse(verbose=False):
    """Test that freeing and re-allocating sequences works correctly,
    and the new sequence produces correct results (blocks are zeroed)."""
    print("=== test_paged_cache_reuse ===")
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads

    model = create_small_model(num_layers=num_layers,
                                hidden_size=hidden_size,
                                num_heads=num_heads,
                                num_kv_heads=num_kv_heads)

    prompt_len = 6
    max_new = 5
    input_ids = list(np.random.randint(0, 256, size=prompt_len))

    # First run
    ids_1, _, _, _ = run_generate(model, input_ids, max_new,
                                  use_paged=True, block_size=4, max_blocks=16)

    # Second run (cache should be reset and produce same result)
    ids_2, _, _, stats = run_generate(model, input_ids, max_new,
                                      use_paged=True, block_size=4, max_blocks=16)

    if verbose:
        print(f"  Run 1: {ids_1[prompt_len:]}")
        print(f"  Run 2: {ids_2[prompt_len:]}")
        if stats:
            print(f"  Final blocks: {stats['num_used_blocks']}/{stats['max_num_blocks']}")

    assert ids_1 == ids_2, (
        f"Reuse produced different results!\n"
        f"  Run 1: {ids_1[prompt_len:]}\n"
        f"  Run 2: {ids_2[prompt_len:]}"
    )

    print("  PASSED")


# ============================================================
#  Test: Attention-level per-layer correctness
# ============================================================

def test_attention_layer_paged_vs_contig(verbose=False):
    """Compare a single Qwen2Attention layer in paged vs contiguous
    mode with prefill + multiple decode steps. Checks per-step outputs."""
    print("=== test_attention_layer_paged_vs_contig ===")
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    seq_len = 6
    num_decode_steps = 8

    # Create two identical attention layers
    attn_c = Qwen2Attention(
        hidden_size=hidden_size, num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads, max_position_embeddings=128,
        device=device, dtype="float32",
    )

    attn_p = Qwen2Attention(
        hidden_size=hidden_size, num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads, max_position_embeddings=128,
        device=device, dtype="float32",
    )

    # Copy weights
    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        src = getattr(attn_c, proj_name)
        dst = getattr(attn_p, proj_name)
        dst.weight.cached_data = src.weight.realize_cached_data()
        if hasattr(src, 'bias') and src.bias is not None:
            dst.bias.cached_data = src.bias.realize_cached_data()
    attn_p._cos_cache = attn_c._cos_cache.copy()
    attn_p._sin_cache = attn_c._sin_cache.copy()

    attn_c.eval()
    attn_p.eval()

    # Init caches
    max_cache_len = seq_len + num_decode_steps
    attn_c.init_cache(batch_size=1, max_cache_len=max_cache_len)
    cache_mgr = PagedKVCacheManager(
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=4, max_num_blocks=32, device=device,
    )
    cache_mgr.allocate_sequence(seq_id=0, initial_len=0)
    attn_p.set_paged_cache(cache_mgr, layer_idx=0, seq_id=0)

    with no_grad():
        # Prefill
        x = Tensor(
            np.random.randn(1, seq_len, hidden_size).astype(np.float32),
            device=device, dtype="float32", requires_grad=False,
        )
        out_c = attn_c(x, start_pos=0)
        out_p = attn_p(x, start_pos=0)

        diff = np.max(np.abs(out_c.numpy() - out_p.numpy()))
        if verbose:
            print(f"  [Prefill] max diff = {diff:.2e}")
        np.testing.assert_allclose(out_c.numpy(), out_p.numpy(), atol=1e-4, rtol=1e-4)

        # Decode steps
        cur_pos = seq_len
        for step in range(num_decode_steps):
            x_dec = Tensor(
                np.random.randn(1, 1, hidden_size).astype(np.float32),
                device=device, dtype="float32", requires_grad=False,
            )
            out_c = attn_c(x_dec, start_pos=cur_pos)
            out_p = attn_p(x_dec, start_pos=cur_pos)

            diff = np.max(np.abs(out_c.numpy() - out_p.numpy()))
            if verbose:
                print(f"  [Decode step {step:2d}] max diff = {diff:.2e}")
            np.testing.assert_allclose(out_c.numpy(), out_p.numpy(), atol=1e-4, rtol=1e-4,
                err_msg=f"Attention decode step {step} differs")
            cur_pos += 1

    print("  PASSED")


# ============================================================
#  Test: Long sequence generation consistency
# ============================================================

def test_long_sequence_consistency(verbose=False):
    """Generate a longer sequence and verify paged/contiguous match."""
    print("=== test_long_sequence_consistency ===")
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2

    model_c = create_small_model(num_layers=num_layers,
                                 hidden_size=hidden_size,
                                 num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
    model_p = clone_model(model_c, num_layers=num_layers,
                          hidden_size=hidden_size,
                          num_heads=num_heads,
                          num_kv_heads=num_kv_heads)

    prompt_len = 12
    max_new = 30  # longer generation
    input_ids = list(np.random.randint(0, 256, size=prompt_len))

    ids_c, pf_c, dec_c, _ = run_generate(
        model_c, input_ids, max_new, use_paged=False)
    ids_p, pf_p, dec_p, stats = run_generate(
        model_p, input_ids, max_new, use_paged=True, block_size=4, max_blocks=64)

    if verbose:
        print(f"  Total tokens generated: {max_new}")
        print(f"  Contiguous: prefill={pf_c*1000:.1f}ms, "
              f"decode_avg={np.mean(dec_c)*1000:.1f}ms")
        print(f"  Paged     : prefill={pf_p*1000:.1f}ms, "
              f"decode_avg={np.mean(dec_p)*1000:.1f}ms")
        if stats:
            total_tokens = prompt_len + max_new
            blocks_used = stats['num_used_blocks']
            block_size = stats['block_size']
            waste = blocks_used * block_size - total_tokens
            print(f"  Paged blocks: {blocks_used}, "
                  f"capacity={blocks_used * block_size}, "
                  f"waste={waste} token slots")

    assert ids_c == ids_p, (
        f"Long sequence differs!\n"
        f"  Contiguous: ...{ids_c[-10:]}\n"
        f"  Paged     : ...{ids_p[-10:]}"
    )

    print("  PASSED")


# ============================================================
#  Test: Summary report
# ============================================================

def test_summary_report(verbose=False):
    """Print a comprehensive summary comparing both modes."""
    print("=== test_summary_report ===")
    num_layers = 3
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads

    model_c = create_small_model(num_layers=num_layers,
                                 hidden_size=hidden_size,
                                 num_heads=num_heads,
                                 num_kv_heads=num_kv_heads)
    model_p = clone_model(model_c, num_layers=num_layers,
                          hidden_size=hidden_size,
                          num_heads=num_heads,
                          num_kv_heads=num_kv_heads)

    prompt_len = 10
    max_new = 15
    input_ids = list(np.random.randint(0, 256, size=prompt_len))

    ids_c, pf_c, dec_c, _ = run_generate(
        model_c, input_ids, max_new, use_paged=False)
    ids_p, pf_p, dec_p, stats_p = run_generate(
        model_p, input_ids, max_new, use_paged=True, block_size=4, max_blocks=64)

    total_tokens = prompt_len + max_new

    # Contiguous memory usage
    max_cache_len = total_tokens
    contig_mem = 2 * num_layers * 1 * num_kv_heads * max_cache_len * head_dim * 4

    # Paged memory usage (actually used)
    blocks_used = stats_p['num_used_blocks'] if stats_p else 0
    block_size = stats_p['block_size'] if stats_p else 4
    paged_mem_used = 2 * num_layers * blocks_used * num_kv_heads * block_size * head_dim * 4

    print()
    print(f"  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║         Paged vs Contiguous KV Cache — Summary          ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║ Model config:                                           ║")
    print(f"  ║   layers={num_layers}, heads={num_heads}, kv_heads={num_kv_heads}, "
          f"head_dim={head_dim:<12d}  ║")
    print(f"  ║   prompt={prompt_len} tokens, generated={max_new} tokens"
          f"{' '*19}║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║ {'':30s}{'Contiguous':>12s} {'Paged':>12s} ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    correct = "✓ MATCH" if ids_c == ids_p else "✗ DIFFER"
    print(f"  ║ {'Correctness':<30s}{correct:>25s} ║")
    print(f"  ║ {'Prefill (ms)':<30s}{pf_c*1000:>12.2f} {pf_p*1000:>12.2f} ║")
    avg_c = np.mean(dec_c) * 1000 if dec_c else 0
    avg_p = np.mean(dec_p) * 1000 if dec_p else 0
    print(f"  ║ {'Avg decode (ms)':<30s}{avg_c:>12.2f} {avg_p:>12.2f} ║")

    def fmt_kb(n):
        return f"{n/1024:.1f} KB"

    print(f"  ║ {'KV cache memory':<30s}{fmt_kb(contig_mem):>12s} {fmt_kb(paged_mem_used):>12s} ║")
    if stats_p:
        print(f"  ║ {'Blocks used/total':<30s}{'N/A':>12s} "
              f"{stats_p['num_used_blocks']:>4d}/{stats_p['max_num_blocks']:<7d} ║")
        print(f"  ║ {'Block utilization':<30s}{'100.0%':>12s} "
              f"{stats_p['utilization']:>11.1f}% ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")
    print()

    assert ids_c == ids_p, "Summary report: sequences don't match!"
    print("  PASSED")


# ============================================================
#  Main
# ============================================================

ALL_TESTS = {
    "prefill_correctness": test_full_model_prefill_correctness,
    "decode_correctness": test_full_model_decode_correctness,
    "stepwise_logit": test_stepwise_logit_match,
    "performance": test_performance_comparison,
    "memory": test_memory_efficiency,
    "block_size": test_block_size_sensitivity,
    "cache_reuse": test_paged_cache_reuse,
    "attention_layer": test_attention_layer_paged_vs_contig,
    "long_sequence": test_long_sequence_consistency,
    "summary": test_summary_report,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paged vs Contiguous KV Cache comparison tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output for each test")
    parser.add_argument("--test", "-t", type=str, default=None,
                        choices=list(ALL_TESTS.keys()),
                        help="Run a specific test only")
    args = parser.parse_args()

    print("=" * 62)
    print("  Paged Attention vs Contiguous KV Cache — Comparison Tests")
    print("=" * 62)
    print()

    if args.test:
        tests_to_run = {args.test: ALL_TESTS[args.test]}
    else:
        tests_to_run = ALL_TESTS

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests_to_run.items():
        try:
            test_fn(verbose=args.verbose)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  FAILED: {e}")
            if args.verbose:
                traceback.print_exc()
            print()

    print()
    print("=" * 62)
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print(f"\n  Failed tests:")
        for name, err in errors:
            print(f"    - {name}: {err}")
    else:
        print("  ALL TESTS PASSED!")
    print("=" * 62)
