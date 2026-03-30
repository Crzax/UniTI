"""Batch Inference Tests for UniTi Framework.

Tests cover:
  1. Batch paged cache allocation: multiple sequences share the block pool.
  2. Batch prefill correctness: each sequence gets correct KV cache independently.
  3. Batch decode correctness: batch decode matches individual single-sequence decode.
  4. Per-sequence start_pos: different sequences at different positions.
  5. Variable-length prompts: different prompt lengths in the same batch.
  6. EOS handling: sequences that finish early are handled properly.
  7. Full model batch inference: small model end-to-end test.
  8. (Optional) Real model batch inference: DeepSeek-R1-Distill-Qwen-1.5B.

Usage:
    # Run unit tests only (no model required):
    python3.10 tests/test_batch_inference.py

    # Run with verbose output:
    python3.10 tests/test_batch_inference.py -v

    # Also run real model test (requires model files):
    python3.10 tests/test_batch_inference.py --real-model

    # Run specific test:
    python3.10 tests/test_batch_inference.py -t batch_allocation
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
from uniti.backend_ndarray.ndarray import NDArray

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

    dst_model.model.embed_tokens.weight.cached_data = \
        src_model.model.embed_tokens.weight.realize_cached_data()

    dst_model.lm_head.weight.cached_data = \
        src_model.lm_head.weight.realize_cached_data()

    for i in range(num_layers):
        src_layer = src_model.model.layers[i]
        dst_layer = dst_model.model.layers[i]

        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            src_proj = getattr(src_layer.self_attn, proj_name)
            dst_proj = getattr(dst_layer.self_attn, proj_name)
            dst_proj.weight.cached_data = src_proj.weight.realize_cached_data()
            if hasattr(src_proj, 'bias') and src_proj.bias is not None:
                dst_proj.bias.cached_data = src_proj.bias.realize_cached_data()

        dst_layer.self_attn._cos_cache = src_layer.self_attn._cos_cache.copy()
        dst_layer.self_attn._sin_cache = src_layer.self_attn._sin_cache.copy()

        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            src_proj = getattr(src_layer.mlp, proj_name)
            dst_proj = getattr(dst_layer.mlp, proj_name)
            dst_proj.weight.cached_data = src_proj.weight.realize_cached_data()

        dst_layer.input_layernorm.weight.cached_data = \
            src_layer.input_layernorm.weight.realize_cached_data()
        dst_layer.post_attention_layernorm.weight.cached_data = \
            src_layer.post_attention_layernorm.weight.realize_cached_data()

    dst_model.model.norm.weight.cached_data = \
        src_model.model.norm.weight.realize_cached_data()

    return dst_model


def run_single_generate(model, input_ids, max_new_tokens,
                        block_size=4, max_blocks=64):
    """Run single-sequence generation with paged cache. Returns generated ids."""
    prompt_len = len(input_ids)
    model.reset_cache()

    cache_mgr = model.init_paged_cache(
        block_size=block_size,
        max_num_blocks=max_blocks,
        seq_id=0,
        initial_len=0,
    )

    ids_np = np.array([input_ids], dtype=np.float32)

    with no_grad():
        ids_tensor = Tensor(ids_np, device=device, dtype="float32", requires_grad=False)
        logits = model(ids_tensor, start_pos=0, last_only=True)
        logits_np = logits.numpy()

    next_tok = int(np.argmax(logits_np[0, -1, :]))
    generated = list(input_ids) + [next_tok]
    cur_pos = prompt_len

    for step in range(1, max_new_tokens):
        with no_grad():
            tok_tensor = Tensor(
                np.array([[next_tok]], dtype=np.float32),
                device=device, dtype="float32", requires_grad=False,
            )
            logits = model(tok_tensor, start_pos=cur_pos, last_only=True)
            logits_np = logits.numpy()
        cur_pos += 1
        next_tok = int(np.argmax(logits_np[0, -1, :]))
        generated.append(next_tok)

    return generated


def run_batch_generate(model, batch_input_ids, max_new_tokens,
                       block_size=4, max_blocks=128):
    """Run batch generation with paged cache. Returns list of generated id lists."""
    batch_size = len(batch_input_ids)
    prompt_lens = [len(ids) for ids in batch_input_ids]

    model.reset_cache()

    seq_ids = list(range(batch_size))
    cache_mgr = model.init_paged_cache(
        block_size=block_size,
        max_num_blocks=max_blocks,
        seq_id=seq_ids,
        initial_len=0,
    )

    # Phase 1: Prefill each sequence individually
    first_logits = []
    with no_grad():
        for b in range(batch_size):
            prompt = batch_input_ids[b]
            ids_np = np.array([prompt], dtype=np.float32)

            for layer in model.model.layers:
                layer.self_attn._paged_seq_ids = [seq_ids[b]]
                layer.self_attn._paged_seq_id = seq_ids[b]

            ids_tensor = Tensor(ids_np, device=device, dtype="float32", requires_grad=False)
            logits = model(ids_tensor, start_pos=0, last_only=True)
            logits_np = logits.numpy()
            first_logits.append(logits_np[0, -1, :])

    # Restore full seq_ids
    for layer in model.model.layers:
        layer.self_attn._paged_seq_ids = seq_ids
        layer.self_attn._paged_seq_id = seq_ids[0]

    # Sample first tokens (greedy)
    next_toks = [int(np.argmax(first_logits[b])) for b in range(batch_size)]

    generated = [list(batch_input_ids[b]) + [next_toks[b]] for b in range(batch_size)]
    cur_pos = list(prompt_lens)

    # Phase 2: Batch decode
    for step in range(1, max_new_tokens):
        with no_grad():
            tok_np = np.array([[next_toks[b]] for b in range(batch_size)], dtype=np.float32)
            tok_tensor = Tensor(
                tok_np, device=device, dtype="float32", requires_grad=False
            )
            logits = model(tok_tensor, start_pos=cur_pos, last_only=True)
            logits_np = logits.numpy()

        for b in range(batch_size):
            cur_pos[b] += 1
            last_logits = logits_np[b, -1, :]
            next_tok = int(np.argmax(last_logits))
            next_toks[b] = next_tok
            generated[b].append(next_tok)

    return generated


# ============================================================
#  Test 1: Batch Paged Cache Allocation
# ============================================================

def test_batch_allocation(verbose=False):
    """Test that multiple sequences can be allocated in paged cache."""
    print("=== test_batch_allocation ===")
    mgr = PagedKVCacheManager(
        num_layers=2, num_kv_heads=4, head_dim=64,
        block_size=4, max_num_blocks=20, device=device,
    )

    # Allocate 3 sequences with different initial lengths
    mgr.allocate_sequence(seq_id=0, initial_len=5)   # 2 blocks
    mgr.allocate_sequence(seq_id=1, initial_len=10)  # 3 blocks
    mgr.allocate_sequence(seq_id=2, initial_len=3)   # 1 block

    assert mgr.num_used_blocks == 6  # 2 + 3 + 1
    assert mgr.num_free_blocks == 14
    assert len(mgr.page_tables) == 3
    assert mgr.seq_lengths[0] == 5
    assert mgr.seq_lengths[1] == 10
    assert mgr.seq_lengths[2] == 3

    if verbose:
        stats = mgr.get_cache_stats()
        print(f"  Blocks used: {stats['num_used_blocks']}/{stats['max_num_blocks']}")
        for sid in sorted(stats['sequences']):
            s = stats['sequences'][sid]
            print(f"  Seq {sid}: length={s['length']}, blocks={s['num_blocks']}")

    # Free one sequence
    mgr.free_sequence(seq_id=1)
    assert mgr.num_free_blocks == 17

    print("  PASSED")


# ============================================================
#  Test 2: Batch Init Paged Cache via Model API
# ============================================================

def test_batch_model_init(verbose=False):
    """Test model.init_paged_cache with a list of seq_ids."""
    print("=== test_batch_model_init ===")
    model = create_small_model()

    seq_ids = [0, 1, 2]
    cache_mgr = model.init_paged_cache(
        block_size=4,
        max_num_blocks=32,
        seq_id=seq_ids,
        initial_len=0,
    )

    # All 3 sequences should be allocated
    assert len(cache_mgr.page_tables) == 3
    for sid in seq_ids:
        assert sid in cache_mgr.page_tables

    # Each attention layer should have all 3 seq_ids
    for layer in model.model.layers:
        assert layer.self_attn._paged_seq_ids == seq_ids

    if verbose:
        print(f"  Allocated {len(seq_ids)} sequences in paged cache")
        print(f"  Each layer has seq_ids: {model.model.layers[0].self_attn._paged_seq_ids}")

    print("  PASSED")


# ============================================================
#  Test 3: Batch vs Single Prefill Correctness
# ============================================================

def test_batch_vs_single_prefill(verbose=False):
    """Verify that prefill in batch mode matches single-sequence prefill."""
    print("=== test_batch_vs_single_prefill ===")
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads

    model = create_small_model(num_layers=num_layers, hidden_size=hidden_size,
                                num_heads=num_heads, num_kv_heads=num_kv_heads)

    # Two prompts with the same length (simplest case)
    prompt_len = 6
    prompt_a = list(np.random.randint(0, 256, size=prompt_len))
    prompt_b = list(np.random.randint(0, 256, size=prompt_len))

    # Run single sequence for each
    model_a = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                          num_heads=num_heads, num_kv_heads=num_kv_heads)
    model_b = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                          num_heads=num_heads, num_kv_heads=num_kv_heads)

    # Single seq: prompt A
    model_a.reset_cache()
    model_a.init_paged_cache(block_size=4, max_num_blocks=32, seq_id=0, initial_len=0)
    with no_grad():
        inp_a = Tensor(np.array([prompt_a], dtype=np.float32), device=device,
                       dtype="float32", requires_grad=False)
        logits_a = model_a(inp_a, start_pos=0, last_only=True).numpy()

    # Single seq: prompt B
    model_b.reset_cache()
    model_b.init_paged_cache(block_size=4, max_num_blocks=32, seq_id=0, initial_len=0)
    with no_grad():
        inp_b = Tensor(np.array([prompt_b], dtype=np.float32), device=device,
                       dtype="float32", requires_grad=False)
        logits_b = model_b(inp_b, start_pos=0, last_only=True).numpy()

    # Batch: both prompts in separate prefill passes (like generate_batch does)
    model_batch = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                              num_heads=num_heads, num_kv_heads=num_kv_heads)
    model_batch.reset_cache()
    cache_mgr = model_batch.init_paged_cache(
        block_size=4, max_num_blocks=64, seq_id=[0, 1], initial_len=0)

    batch_logits = []
    with no_grad():
        for b, prompt in enumerate([prompt_a, prompt_b]):
            for layer in model_batch.model.layers:
                layer.self_attn._paged_seq_ids = [b]
                layer.self_attn._paged_seq_id = b
            inp = Tensor(np.array([prompt], dtype=np.float32), device=device,
                         dtype="float32", requires_grad=False)
            logits = model_batch(inp, start_pos=0, last_only=True).numpy()
            batch_logits.append(logits)

    # Compare
    max_diff_a = np.max(np.abs(logits_a - batch_logits[0]))
    max_diff_b = np.max(np.abs(logits_b - batch_logits[1]))

    if verbose:
        print(f"  Max diff prompt A: {max_diff_a:.2e}")
        print(f"  Max diff prompt B: {max_diff_b:.2e}")

    np.testing.assert_allclose(logits_a, batch_logits[0], atol=1e-4, rtol=1e-4,
        err_msg="Prefill logits for prompt A differ between single and batch mode!")
    np.testing.assert_allclose(logits_b, batch_logits[1], atol=1e-4, rtol=1e-4,
        err_msg="Prefill logits for prompt B differ between single and batch mode!")

    print("  PASSED")


# ============================================================
#  Test 4: Batch Decode vs Single Decode Correctness
# ============================================================

def test_batch_vs_single_decode(verbose=False):
    """Run full generation in batch mode and compare with individual single-sequence runs."""
    print("=== test_batch_vs_single_decode ===")
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2

    model = create_small_model(num_layers=num_layers, hidden_size=hidden_size,
                                num_heads=num_heads, num_kv_heads=num_kv_heads)

    # Two prompts with same length
    prompt_len = 8
    max_new = 10
    prompt_a = list(np.random.randint(0, 256, size=prompt_len))
    prompt_b = list(np.random.randint(0, 256, size=prompt_len))

    # Single runs
    model_a = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                          num_heads=num_heads, num_kv_heads=num_kv_heads)
    model_b = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                          num_heads=num_heads, num_kv_heads=num_kv_heads)
    ids_a = run_single_generate(model_a, prompt_a, max_new, block_size=4, max_blocks=32)
    ids_b = run_single_generate(model_b, prompt_b, max_new, block_size=4, max_blocks=32)

    # Batch run
    model_batch = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                              num_heads=num_heads, num_kv_heads=num_kv_heads)
    batch_ids = run_batch_generate(model_batch, [prompt_a, prompt_b], max_new,
                                   block_size=4, max_blocks=64)

    if verbose:
        print(f"  Single A: {ids_a[prompt_len:]}")
        print(f"  Batch  A: {batch_ids[0][prompt_len:]}")
        print(f"  Single B: {ids_b[prompt_len:]}")
        print(f"  Batch  B: {batch_ids[1][prompt_len:]}")

    assert ids_a == batch_ids[0], (
        f"Sequence A differs!\n"
        f"  Single: {ids_a[prompt_len:]}\n"
        f"  Batch:  {batch_ids[0][prompt_len:]}"
    )
    assert ids_b == batch_ids[1], (
        f"Sequence B differs!\n"
        f"  Single: {ids_b[prompt_len:]}\n"
        f"  Batch:  {batch_ids[1][prompt_len:]}"
    )

    print("  PASSED")


# ============================================================
#  Test 5: Variable-Length Prompts in Batch
# ============================================================

def test_variable_length_prompts(verbose=False):
    """Test batch with prompts of different lengths."""
    print("=== test_variable_length_prompts ===")
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2

    model = create_small_model(num_layers=num_layers, hidden_size=hidden_size,
                                num_heads=num_heads, num_kv_heads=num_kv_heads)

    # Three prompts with different lengths
    prompt_a = list(np.random.randint(0, 256, size=4))
    prompt_b = list(np.random.randint(0, 256, size=8))
    prompt_c = list(np.random.randint(0, 256, size=12))
    max_new = 5

    # Single runs
    model_a = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                          num_heads=num_heads, num_kv_heads=num_kv_heads)
    model_b = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                          num_heads=num_heads, num_kv_heads=num_kv_heads)
    model_c = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                          num_heads=num_heads, num_kv_heads=num_kv_heads)
    ids_a = run_single_generate(model_a, prompt_a, max_new, block_size=4, max_blocks=32)
    ids_b = run_single_generate(model_b, prompt_b, max_new, block_size=4, max_blocks=32)
    ids_c = run_single_generate(model_c, prompt_c, max_new, block_size=4, max_blocks=32)

    # Batch run
    model_batch = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                              num_heads=num_heads, num_kv_heads=num_kv_heads)
    batch_ids = run_batch_generate(model_batch, [prompt_a, prompt_b, prompt_c], max_new,
                                   block_size=4, max_blocks=128)

    if verbose:
        for i, (single_ids, label) in enumerate(
            [(ids_a, "A(4)"), (ids_b, "B(8)"), (ids_c, "C(12)")]):
            pl = [4, 8, 12][i]
            print(f"  Single {label}: {single_ids[pl:]}")
            print(f"  Batch  {label}: {batch_ids[i][pl:]}")

    assert ids_a == batch_ids[0], (
        f"Sequence A (len=4) differs!\n"
        f"  Single: {ids_a[4:]}\n"
        f"  Batch:  {batch_ids[0][4:]}"
    )
    assert ids_b == batch_ids[1], (
        f"Sequence B (len=8) differs!\n"
        f"  Single: {ids_b[8:]}\n"
        f"  Batch:  {batch_ids[1][8:]}"
    )
    assert ids_c == batch_ids[2], (
        f"Sequence C (len=12) differs!\n"
        f"  Single: {ids_c[12:]}\n"
        f"  Batch:  {batch_ids[2][12:]}"
    )

    print("  PASSED")


# ============================================================
#  Test 6: Paged Cache Memory Sharing
# ============================================================

def test_cache_memory_sharing(verbose=False):
    """Test that batch sequences share physical block pool efficiently."""
    print("=== test_cache_memory_sharing ===")
    num_kv_heads = 2
    head_dim = 8
    block_size = 4
    max_blocks = 20

    mgr = PagedKVCacheManager(
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=block_size, max_num_blocks=max_blocks, device=device,
    )

    # Allocate 4 sequences
    for sid in range(4):
        mgr.allocate_sequence(seq_id=sid, initial_len=0)

    # Append different amounts of data
    lengths = [3, 7, 5, 1]
    seq_data = {}
    for sid, length in enumerate(lengths):
        k = np.random.randn(num_kv_heads, length, head_dim).astype(np.float32)
        v = np.random.randn(num_kv_heads, length, head_dim).astype(np.float32)
        seq_data[sid] = (k, v)
        mgr.append_kv(seq_id=sid, layer_idx=0,
                      k_data=NDArray(k, device=device), v_data=NDArray(v, device=device))

    # Expected blocks: ceil(3/4) + ceil(7/4) + ceil(5/4) + ceil(1/4) = 1 + 2 + 2 + 1 = 6
    expected_blocks = sum((l + block_size - 1) // block_size for l in lengths)
    assert mgr.num_used_blocks == expected_blocks, \
        f"Expected {expected_blocks} blocks, got {mgr.num_used_blocks}"
    assert mgr.num_free_blocks == max_blocks - expected_blocks

    # Verify each sequence's data
    for sid in range(4):
        k_out, v_out = mgr.gather_kv(seq_id=sid, layer_idx=0)
        k_expected, v_expected = seq_data[sid]
        np.testing.assert_allclose(k_out, k_expected, atol=1e-6)
        np.testing.assert_allclose(v_out, v_expected, atol=1e-6)

    if verbose:
        stats = mgr.get_cache_stats()
        print(f"  Total blocks: {stats['num_used_blocks']}/{stats['max_num_blocks']}")
        for sid in range(4):
            s = stats['sequences'][sid]
            print(f"  Seq {sid}: length={s['length']}, blocks={s['num_blocks']}")

    # Free seq 1 and verify others are intact
    mgr.free_sequence(seq_id=1)
    for sid in [0, 2, 3]:
        k_out, v_out = mgr.gather_kv(seq_id=sid, layer_idx=0)
        k_expected, v_expected = seq_data[sid]
        np.testing.assert_allclose(k_out, k_expected, atol=1e-6)

    print("  PASSED")


# ============================================================
#  Test 7: Per-Sequence Position Tracking
# ============================================================

def test_per_sequence_position(verbose=False):
    """Test that attention correctly handles per-sequence start_pos in batch decode."""
    print("=== test_per_sequence_position ===")
    hidden_size = 32
    num_heads = 2
    num_kv_heads = 2
    head_dim = hidden_size // num_heads

    attn = Qwen2Attention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=128,
        device=device, dtype="float32",
    )
    attn.eval()

    # Set up paged cache for 2 sequences
    cache_mgr = PagedKVCacheManager(
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=4, max_num_blocks=20, device=device,
    )
    cache_mgr.allocate_sequence(seq_id=0, initial_len=0)
    cache_mgr.allocate_sequence(seq_id=1, initial_len=0)
    attn.set_paged_cache(cache_mgr, layer_idx=0, seq_id=[0, 1])

    with no_grad():
        # Prefill seq 0 with 3 tokens
        for layer_id in [0]:
            attn._paged_seq_ids = [0]
        x0 = Tensor(
            np.random.randn(1, 3, hidden_size).astype(np.float32),
            device=device, dtype="float32", requires_grad=False,
        )
        out0 = attn(x0, start_pos=0)

        # Prefill seq 1 with 5 tokens
        attn._paged_seq_ids = [1]
        x1 = Tensor(
            np.random.randn(1, 5, hidden_size).astype(np.float32),
            device=device, dtype="float32", requires_grad=False,
        )
        out1 = attn(x1, start_pos=0)

        # Verify KV cache lengths
        assert cache_mgr.seq_lengths[0] == 3
        assert cache_mgr.seq_lengths[1] == 5

        # Batch decode with different start_pos
        attn._paged_seq_ids = [0, 1]
        x_decode = Tensor(
            np.random.randn(2, 1, hidden_size).astype(np.float32),
            device=device, dtype="float32", requires_grad=False,
        )
        out_batch = attn(x_decode, start_pos=[3, 5])

        assert out_batch.shape == (2, 1, hidden_size)

        # After decode, seq 0 should have 4 tokens, seq 1 should have 6
        assert cache_mgr.seq_lengths[0] == 4
        assert cache_mgr.seq_lengths[1] == 6

    if verbose:
        print(f"  Seq 0 cache length: {cache_mgr.seq_lengths[0]} (expected 4)")
        print(f"  Seq 1 cache length: {cache_mgr.seq_lengths[1]} (expected 6)")

    print("  PASSED")


# ============================================================
#  Test 8: Batch Size Scaling
# ============================================================

def test_batch_size_scaling(verbose=False):
    """Test batch inference with various batch sizes (1, 2, 4)."""
    print("=== test_batch_size_scaling ===")
    num_layers = 2
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2

    model = create_small_model(num_layers=num_layers, hidden_size=hidden_size,
                                num_heads=num_heads, num_kv_heads=num_kv_heads)

    prompt_len = 6
    max_new = 5

    for batch_size in [1, 2, 4]:
        prompts = [list(np.random.randint(0, 256, size=prompt_len))
                   for _ in range(batch_size)]

        # Single runs for reference
        single_ids = []
        for prompt in prompts:
            m = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                            num_heads=num_heads, num_kv_heads=num_kv_heads)
            ids = run_single_generate(m, prompt, max_new, block_size=4, max_blocks=32)
            single_ids.append(ids)

        # Batch run
        m_batch = clone_model(model, num_layers=num_layers, hidden_size=hidden_size,
                              num_heads=num_heads, num_kv_heads=num_kv_heads)
        batch_ids = run_batch_generate(m_batch, prompts, max_new,
                                       block_size=4, max_blocks=32 * batch_size)

        all_match = True
        for b in range(batch_size):
            if single_ids[b] != batch_ids[b]:
                all_match = False
                if verbose:
                    print(f"  batch_size={batch_size}, seq {b}: MISMATCH")
                    print(f"    Single: {single_ids[b][prompt_len:]}")
                    print(f"    Batch:  {batch_ids[b][prompt_len:]}")

        assert all_match, f"batch_size={batch_size}: some sequences don't match!"

        if verbose:
            print(f"  batch_size={batch_size}: all {batch_size} sequences match ✓")

    print("  PASSED")


# ============================================================
#  Test 9: (Optional) Real Model Batch Inference
# ============================================================

def test_real_model_batch(verbose=False):
    """End-to-end batch inference with real DeepSeek-R1-Distill-Qwen-1.5B model.
    
    This test requires the model files and is skipped by default.
    Use --real-model flag to enable.
    """
    print("=== test_real_model_batch ===")
    import json

    model_path = ""
    if not os.path.exists(model_path):
        print("  SKIPPED (model not found)")
        return

    from uniti.safetensors_loader import load_safetensors_sharded
    from uniti.tokenizer import UniTITokenizer

    # Load config
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)

    model = Qwen2ForCausalLM(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        intermediate_size=config["intermediate_size"],
        rms_norm_eps=config.get("rms_norm_eps", 1e-6),
        max_position_embeddings=config.get("max_position_embeddings", 131072),
        rope_theta=config.get("rope_theta", 10000.0),
        tie_word_embeddings=config.get("tie_word_embeddings", False),
        device=device,
        dtype="float32",
    )

    print("  Loading weights...")
    t0 = time.time()
    state_dict = load_safetensors_sharded(model_path)

    # Load weights (inline)
    dtype = model.dtype
    def _set_param(param, np_arr):
        param.cached_data = Tensor._array_from_numpy(
            np_arr.astype(np.float32), device=device, dtype=dtype)
    def _set_linear(linear, weight_np, bias_np=None):
        _set_param(linear.weight, weight_np.T)
        if bias_np is not None and linear.bias is not None:
            _set_param(linear.bias, bias_np.reshape(1, -1))

    _set_param(model.model.embed_tokens.weight, state_dict["model.embed_tokens.weight"])
    model.model.embed_tokens._invalidate_cache()
    if "lm_head.weight" in state_dict:
        _set_linear(model.lm_head, state_dict["lm_head.weight"])
    else:
        _set_linear(model.lm_head, state_dict["model.embed_tokens.weight"])
    _set_param(model.model.norm.weight, state_dict["model.norm.weight"])

    num_layers = len(model.model.layers)
    for i in range(num_layers):
        p = f"model.layers.{i}"
        layer = model.model.layers[i]
        _set_linear(layer.self_attn.q_proj, state_dict[f"{p}.self_attn.q_proj.weight"],
                     state_dict[f"{p}.self_attn.q_proj.bias"])
        _set_linear(layer.self_attn.k_proj, state_dict[f"{p}.self_attn.k_proj.weight"],
                     state_dict[f"{p}.self_attn.k_proj.bias"])
        _set_linear(layer.self_attn.v_proj, state_dict[f"{p}.self_attn.v_proj.weight"],
                     state_dict[f"{p}.self_attn.v_proj.bias"])
        _set_linear(layer.self_attn.o_proj, state_dict[f"{p}.self_attn.o_proj.weight"])
        _set_linear(layer.mlp.gate_proj, state_dict[f"{p}.mlp.gate_proj.weight"])
        _set_linear(layer.mlp.up_proj, state_dict[f"{p}.mlp.up_proj.weight"])
        _set_linear(layer.mlp.down_proj, state_dict[f"{p}.mlp.down_proj.weight"])
        _set_param(layer.input_layernorm.weight, state_dict[f"{p}.input_layernorm.weight"])
        _set_param(layer.post_attention_layernorm.weight, state_dict[f"{p}.post_attention_layernorm.weight"])

    del state_dict
    print(f"  Loaded in {time.time()-t0:.1f}s")

    tokenizer = UniTITokenizer.from_pretrained(model_path)
    model.eval()

    # Tokenize batch
    prompts = [
        "What is 2+2?",
        "Hello!",
        "Explain gravity in one sentence.",
    ]
    eos_token_id = config.get("eos_token_id", 151643)

    batch_input_ids = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(chat_text)
        batch_input_ids.append(ids)
        print(f"  Prompt: \"{prompt}\" → {len(ids)} tokens")

    # Single greedy runs for reference
    max_new = 20
    print("  Running single-sequence references...")
    single_results = []
    for b, ids in enumerate(batch_input_ids):
        model.reset_cache()
        cache_mgr = model.init_paged_cache(block_size=16, max_num_blocks=64, seq_id=0, initial_len=0)
        ids_np = np.array([ids], dtype=np.float32)
        generated = list(ids)
        cur_pos = len(ids)

        with no_grad():
            ids_tensor = Tensor(ids_np, device=device, dtype="float32", requires_grad=False)
            logits = model(ids_tensor, start_pos=0, last_only=True)
            logits_np = logits.numpy()
            next_tok = int(np.argmax(logits_np[0, -1, :]))
            generated.append(next_tok)

            for step in range(1, max_new):
                tok_tensor = Tensor(
                    np.array([[next_tok]], dtype=np.float32),
                    device=device, dtype="float32", requires_grad=False)
                logits = model(tok_tensor, start_pos=cur_pos, last_only=True)
                logits_np = logits.numpy()
                cur_pos += 1
                next_tok = int(np.argmax(logits_np[0, -1, :]))
                generated.append(next_tok)
                if next_tok == eos_token_id:
                    break

        single_results.append(generated)

    # Batch run
    print("  Running batch inference...")
    batch_size = len(prompts)
    model.reset_cache()
    seq_ids = list(range(batch_size))
    cache_mgr = model.init_paged_cache(
        block_size=16, max_num_blocks=256, seq_id=seq_ids, initial_len=0)

    first_logits = []
    with no_grad():
        for b in range(batch_size):
            for layer in model.model.layers:
                layer.self_attn._paged_seq_ids = [seq_ids[b]]
                layer.self_attn._paged_seq_id = seq_ids[b]
            ids_np = np.array([batch_input_ids[b]], dtype=np.float32)
            ids_tensor = Tensor(ids_np, device=device, dtype="float32", requires_grad=False)
            logits = model(ids_tensor, start_pos=0, last_only=True)
            first_logits.append(logits.numpy()[0, -1, :])

    for layer in model.model.layers:
        layer.self_attn._paged_seq_ids = seq_ids
        layer.self_attn._paged_seq_id = seq_ids[0]

    next_toks = [int(np.argmax(first_logits[b])) for b in range(batch_size)]
    prompt_lens = [len(ids) for ids in batch_input_ids]
    batch_generated = [list(batch_input_ids[b]) + [next_toks[b]] for b in range(batch_size)]
    cur_pos = list(prompt_lens)
    finished = [next_toks[b] == eos_token_id for b in range(batch_size)]

    for step in range(1, max_new):
        if all(finished):
            break
        with no_grad():
            tok_np = np.array([[next_toks[b]] for b in range(batch_size)], dtype=np.float32)
            tok_tensor = Tensor(tok_np, device=device, dtype="float32", requires_grad=False)
            logits = model(tok_tensor, start_pos=cur_pos, last_only=True)
            logits_np = logits.numpy()

        for b in range(batch_size):
            if finished[b]:
                continue
            cur_pos[b] += 1
            next_tok = int(np.argmax(logits_np[b, -1, :]))
            next_toks[b] = next_tok
            batch_generated[b].append(next_tok)
            if next_tok == eos_token_id:
                finished[b] = True

    # Compare results
    all_match = True
    for b in range(batch_size):
        # Compare up to the shorter length (batch may generate more due to not stopping)
        min_len = min(len(single_results[b]), len(batch_generated[b]))
        single_gen = single_results[b][:min_len]
        batch_gen = batch_generated[b][:min_len]
        match = single_gen == batch_gen
        text_single = tokenizer.decode(single_results[b][prompt_lens[b]:], skip_special_tokens=True)
        text_batch = tokenizer.decode(batch_generated[b][prompt_lens[b]:], skip_special_tokens=True)

        status = "✓" if match else "✗"
        print(f"  Seq {b} [{status}]: \"{prompts[b]}\"")
        if verbose or not match:
            print(f"    Single: {text_single[:100]}")
            print(f"    Batch:  {text_batch[:100]}")
        if not match:
            all_match = False

    assert all_match, "Some sequences don't match between single and batch mode!"
    print("  PASSED")


# ============================================================
#  Main
# ============================================================

ALL_TESTS = {
    "batch_allocation": test_batch_allocation,
    "batch_model_init": test_batch_model_init,
    "batch_vs_single_prefill": test_batch_vs_single_prefill,
    "batch_vs_single_decode": test_batch_vs_single_decode,
    "variable_length_prompts": test_variable_length_prompts,
    "cache_memory_sharing": test_cache_memory_sharing,
    "per_sequence_position": test_per_sequence_position,
    "batch_size_scaling": test_batch_size_scaling,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Inference Tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--test", "-t", type=str, default=None,
                        choices=list(ALL_TESTS.keys()) + ["real_model"],
                        help="Run specific test only")
    parser.add_argument("--real-model", action="store_true",
                        help="Also run the real model batch test")
    args = parser.parse_args()

    print("=" * 62)
    print("  Batch Inference Tests for UniTi Framework")
    print("=" * 62)
    print()

    if args.test:
        if args.test == "real_model":
            tests_to_run = {"real_model": test_real_model_batch}
        else:
            tests_to_run = {args.test: ALL_TESTS[args.test]}
    else:
        tests_to_run = dict(ALL_TESTS)
        if args.real_model:
            tests_to_run["real_model"] = test_real_model_batch

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
