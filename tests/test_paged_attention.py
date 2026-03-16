"""Unit tests for Paged Attention in UniTi framework.

Tests cover:
  1. PagedKVCacheManager: block allocation, write/read, gather, free, stats.
  2. Qwen2Attention integration: paged mode produces same results as contiguous mode.
  3. Edge cases: block boundary crossings, multi-sequence support, out-of-blocks.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

import uniti
from uniti.autograd import Tensor, no_grad
from uniti.nn.paged_attention import PagedKVCacheManager
from uniti.nn.nn_qwen2 import Qwen2Attention, Qwen2ForCausalLM
from uniti.backend_selection import cpu_numpy

np.random.seed(42)
device = cpu_numpy()


def test_basic_allocation():
    """Test basic block allocation and freeing."""
    print("=== test_basic_allocation ===")
    mgr = PagedKVCacheManager(
        num_layers=2, num_kv_heads=4, head_dim=64,
        block_size=4, max_num_blocks=10, device=device,
    )
    
    assert mgr.num_free_blocks == 10
    assert mgr.num_used_blocks == 0
    
    # Allocate sequence with 10 tokens -> needs ceil(10/4) = 3 blocks
    mgr.allocate_sequence(seq_id=0, initial_len=10)
    assert mgr.num_free_blocks == 7
    assert mgr.num_used_blocks == 3
    assert len(mgr.page_tables[0]) == 3
    assert mgr.seq_lengths[0] == 10
    
    # Free it
    mgr.free_sequence(seq_id=0)
    assert mgr.num_free_blocks == 10
    assert 0 not in mgr.page_tables
    
    print("  PASSED")


def test_append_and_gather():
    """Test writing KV pairs and reading them back."""
    print("=== test_append_and_gather ===")
    num_kv_heads = 2
    head_dim = 8
    block_size = 4
    
    mgr = PagedKVCacheManager(
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=block_size, max_num_blocks=10, device=device,
    )
    
    mgr.allocate_sequence(seq_id=0, initial_len=0)
    
    # Append 3 tokens
    k1 = np.random.randn(num_kv_heads, 3, head_dim).astype(np.float32)
    v1 = np.random.randn(num_kv_heads, 3, head_dim).astype(np.float32)
    mgr.append_kv(seq_id=0, layer_idx=0, k_data=k1, v_data=v1)
    
    assert mgr.seq_lengths[0] == 3
    assert len(mgr.page_tables[0]) == 1  # 3 tokens fit in 1 block of size 4
    
    # Gather and verify
    k_out, v_out = mgr.gather_kv(seq_id=0, layer_idx=0)
    assert k_out.shape == (num_kv_heads, 3, head_dim)
    np.testing.assert_allclose(k_out, k1, atol=1e-6)
    np.testing.assert_allclose(v_out, v1, atol=1e-6)
    
    # Append 3 more tokens (crosses block boundary: 3+3=6, needs 2 blocks)
    k2 = np.random.randn(num_kv_heads, 3, head_dim).astype(np.float32)
    v2 = np.random.randn(num_kv_heads, 3, head_dim).astype(np.float32)
    mgr.append_kv(seq_id=0, layer_idx=0, k_data=k2, v_data=v2)
    
    assert mgr.seq_lengths[0] == 6
    assert len(mgr.page_tables[0]) == 2  # 6 tokens need 2 blocks of size 4
    
    # Gather and verify all 6 tokens
    k_out, v_out = mgr.gather_kv(seq_id=0, layer_idx=0)
    expected_k = np.concatenate([k1, k2], axis=1)
    expected_v = np.concatenate([v1, v2], axis=1)
    np.testing.assert_allclose(k_out, expected_k, atol=1e-6)
    np.testing.assert_allclose(v_out, expected_v, atol=1e-6)
    
    print("  PASSED")


def test_multi_layer():
    """Test that per-layer KV caches are independent."""
    print("=== test_multi_layer ===")
    num_layers = 3
    num_kv_heads = 2
    head_dim = 8
    
    mgr = PagedKVCacheManager(
        num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=4, max_num_blocks=20, device=device,
    )
    
    mgr.allocate_sequence(seq_id=0, initial_len=0)
    
    # Write different data to each layer
    layer_data = {}
    for layer_idx in range(num_layers):
        k = np.random.randn(num_kv_heads, 5, head_dim).astype(np.float32)
        v = np.random.randn(num_kv_heads, 5, head_dim).astype(np.float32)
        layer_data[layer_idx] = (k, v)
        mgr.append_kv(seq_id=0, layer_idx=layer_idx, k_data=k, v_data=v)
    
    # Verify each layer has its own data
    for layer_idx in range(num_layers):
        k_out, v_out = mgr.gather_kv(seq_id=0, layer_idx=layer_idx)
        k_expected, v_expected = layer_data[layer_idx]
        np.testing.assert_allclose(k_out, k_expected, atol=1e-6)
        np.testing.assert_allclose(v_out, v_expected, atol=1e-6)
    
    print("  PASSED")


def test_multi_sequence():
    """Test multiple sequences sharing the same block pool."""
    print("=== test_multi_sequence ===")
    num_kv_heads = 2
    head_dim = 8
    
    mgr = PagedKVCacheManager(
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=4, max_num_blocks=10, device=device,
    )
    
    # Allocate 3 sequences
    for sid in range(3):
        mgr.allocate_sequence(seq_id=sid, initial_len=0)
    
    # Write different lengths
    seq_data = {}
    for sid, length in [(0, 3), (1, 7), (2, 1)]:
        k = np.random.randn(num_kv_heads, length, head_dim).astype(np.float32)
        v = np.random.randn(num_kv_heads, length, head_dim).astype(np.float32)
        seq_data[sid] = (k, v)
        mgr.append_kv(seq_id=sid, layer_idx=0, k_data=k, v_data=v)
    
    # Verify each sequence
    for sid in range(3):
        k_out, v_out = mgr.gather_kv(seq_id=sid, layer_idx=0)
        k_expected, v_expected = seq_data[sid]
        np.testing.assert_allclose(k_out, k_expected, atol=1e-6)
    
    # Free sequence 1 and check blocks returned
    blocks_before = mgr.num_free_blocks
    seq1_blocks = len(mgr.page_tables[1])
    mgr.free_sequence(seq_id=1)
    assert mgr.num_free_blocks == blocks_before + seq1_blocks
    
    # Sequence 0 and 2 still intact
    for sid in [0, 2]:
        k_out, v_out = mgr.gather_kv(seq_id=sid, layer_idx=0)
        k_expected, v_expected = seq_data[sid]
        np.testing.assert_allclose(k_out, k_expected, atol=1e-6)
    
    print("  PASSED")


def test_out_of_blocks():
    """Test that allocation fails gracefully when blocks are exhausted."""
    print("=== test_out_of_blocks ===")
    mgr = PagedKVCacheManager(
        num_layers=1, num_kv_heads=2, head_dim=8,
        block_size=4, max_num_blocks=3, device=device,
    )
    
    # Allocate all 3 blocks (12 tokens)
    mgr.allocate_sequence(seq_id=0, initial_len=12)
    assert mgr.num_free_blocks == 0
    
    # Trying to allocate more should fail
    try:
        mgr.allocate_sequence(seq_id=1, initial_len=1)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Out of physical blocks" in str(e) or "Cannot allocate" in str(e)
    
    # Free and re-allocate
    mgr.free_sequence(seq_id=0)
    mgr.allocate_sequence(seq_id=1, initial_len=4)  # This should work now
    assert mgr.num_free_blocks == 2
    
    print("  PASSED")


def test_gather_kv_tensor():
    """Test that gather_kv_tensor returns proper UniTi Tensors."""
    print("=== test_gather_kv_tensor ===")
    num_kv_heads = 2
    head_dim = 8
    
    mgr = PagedKVCacheManager(
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=4, max_num_blocks=10, device=device,
    )
    
    mgr.allocate_sequence(seq_id=0, initial_len=0)
    
    k = np.random.randn(num_kv_heads, 5, head_dim).astype(np.float32)
    v = np.random.randn(num_kv_heads, 5, head_dim).astype(np.float32)
    mgr.append_kv(seq_id=0, layer_idx=0, k_data=k, v_data=v)
    
    k_t, v_t = mgr.gather_kv_tensor(seq_id=0, layer_idx=0)
    
    assert isinstance(k_t, Tensor)
    assert isinstance(v_t, Tensor)
    assert k_t.shape == (1, num_kv_heads, 5, head_dim)
    assert v_t.shape == (1, num_kv_heads, 5, head_dim)
    
    np.testing.assert_allclose(k_t.numpy()[0], k, atol=1e-6)
    np.testing.assert_allclose(v_t.numpy()[0], v, atol=1e-6)
    
    print("  PASSED")


def test_cache_stats():
    """Test cache statistics reporting."""
    print("=== test_cache_stats ===")
    mgr = PagedKVCacheManager(
        num_layers=2, num_kv_heads=4, head_dim=64,
        block_size=16, max_num_blocks=100, device=device,
    )
    
    mgr.allocate_sequence(seq_id=0, initial_len=0)
    k = np.random.randn(4, 50, 64).astype(np.float32)
    v = np.random.randn(4, 50, 64).astype(np.float32)
    mgr.append_kv(seq_id=0, layer_idx=0, k_data=k, v_data=v)
    
    stats = mgr.get_cache_stats()
    assert stats["num_layers"] == 2
    assert stats["block_size"] == 16
    assert stats["max_num_blocks"] == 100
    assert stats["num_sequences"] == 1
    assert stats["sequences"][0]["length"] == 50
    # 50 tokens / 16 block_size = ceil(50/16) = 4 blocks
    assert stats["sequences"][0]["num_blocks"] == 4
    
    print("  PASSED")


def test_reset():
    """Test full cache reset."""
    print("=== test_reset ===")
    mgr = PagedKVCacheManager(
        num_layers=1, num_kv_heads=2, head_dim=8,
        block_size=4, max_num_blocks=10, device=device,
    )
    
    mgr.allocate_sequence(seq_id=0, initial_len=8)
    mgr.allocate_sequence(seq_id=1, initial_len=4)
    assert mgr.num_used_blocks == 3  # 2 + 1
    
    mgr.reset()
    assert mgr.num_free_blocks == 10
    assert len(mgr.page_tables) == 0
    assert len(mgr.seq_layer_lengths) == 0
    
    print("  PASSED")


def test_paged_vs_contiguous_attention():
    """Integration test: paged mode produces same output as contiguous mode.
    
    Creates a small Qwen2Attention, runs the same input through both
    contiguous and paged cache modes, and verifies output matches.
    """
    print("=== test_paged_vs_contiguous_attention ===")
    hidden_size = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    seq_len = 8
    batch_size = 1
    
    # Create two identical attention modules
    attn_contiguous = Qwen2Attention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=128,
        device=device, dtype="float32",
    )
    
    attn_paged = Qwen2Attention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=128,
        device=device, dtype="float32",
    )
    
    # Copy weights from contiguous to paged
    attn_paged.q_proj.weight.cached_data = attn_contiguous.q_proj.weight.realize_cached_data()
    attn_paged.k_proj.weight.cached_data = attn_contiguous.k_proj.weight.realize_cached_data()
    attn_paged.v_proj.weight.cached_data = attn_contiguous.v_proj.weight.realize_cached_data()
    attn_paged.o_proj.weight.cached_data = attn_contiguous.o_proj.weight.realize_cached_data()
    attn_paged.q_proj.bias.cached_data = attn_contiguous.q_proj.bias.realize_cached_data()
    attn_paged.k_proj.bias.cached_data = attn_contiguous.k_proj.bias.realize_cached_data()
    attn_paged.v_proj.bias.cached_data = attn_contiguous.v_proj.bias.realize_cached_data()
    attn_paged._cos_cache = attn_contiguous._cos_cache
    attn_paged._sin_cache = attn_contiguous._sin_cache
    
    # Setup caches
    attn_contiguous.init_cache(batch_size=1, max_cache_len=32)
    
    cache_mgr = PagedKVCacheManager(
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=4, max_num_blocks=20, device=device,
    )
    cache_mgr.allocate_sequence(seq_id=0, initial_len=0)
    attn_paged.set_paged_cache(cache_mgr, layer_idx=0, seq_id=0)
    
    # Create input
    x = Tensor(
        np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32),
        device=device, dtype="float32", requires_grad=False,
    )
    
    attn_contiguous.eval()
    attn_paged.eval()
    
    with no_grad():
        # Prefill
        out_contiguous = attn_contiguous(x, start_pos=0)
        out_paged = attn_paged(x, start_pos=0)
        
        np.testing.assert_allclose(
            out_contiguous.numpy(), out_paged.numpy(),
            atol=1e-4, rtol=1e-4,
            err_msg="Prefill outputs differ between contiguous and paged mode!"
        )
        
        # Decode step
        x_decode = Tensor(
            np.random.randn(batch_size, 1, hidden_size).astype(np.float32),
            device=device, dtype="float32", requires_grad=False,
        )
        
        out_contiguous_decode = attn_contiguous(x_decode, start_pos=seq_len)
        out_paged_decode = attn_paged(x_decode, start_pos=seq_len)
        
        np.testing.assert_allclose(
            out_contiguous_decode.numpy(), out_paged_decode.numpy(),
            atol=1e-4, rtol=1e-4,
            err_msg="Decode outputs differ between contiguous and paged mode!"
        )
    
    print("  PASSED")


def test_incremental_decode_consistency():
    """Test that incremental decode with paged cache is consistent over multiple steps."""
    print("=== test_incremental_decode_consistency ===")
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
    
    cache_mgr = PagedKVCacheManager(
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=4, max_num_blocks=20, device=device,
    )
    cache_mgr.allocate_sequence(seq_id=0, initial_len=0)
    attn.set_paged_cache(cache_mgr, layer_idx=0, seq_id=0)
    
    # Run 10 decode steps
    outputs = []
    with no_grad():
        for pos in range(10):
            x = Tensor(
                np.random.randn(1, 1, hidden_size).astype(np.float32),
                device=device, dtype="float32", requires_grad=False,
            )
            out = attn(x, start_pos=pos)
            outputs.append(out.numpy())
            
            # Verify cache length grows
            assert cache_mgr.seq_lengths[0] == pos + 1
    
    # Verify all outputs have correct shape
    for i, out in enumerate(outputs):
        assert out.shape == (1, 1, hidden_size), f"Step {i}: shape {out.shape}"
    
    # Cache should have 10 tokens across ceil(10/4) = 3 blocks
    assert len(cache_mgr.page_tables[0]) == 3
    
    print("  PASSED")


def test_block_boundary_crossing():
    """Test behavior when tokens span multiple block boundaries."""
    print("=== test_block_boundary_crossing ===")
    num_kv_heads = 2
    head_dim = 8
    block_size = 4
    
    mgr = PagedKVCacheManager(
        num_layers=1, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=block_size, max_num_blocks=10, device=device,
    )
    
    mgr.allocate_sequence(seq_id=0, initial_len=0)
    
    # Append exactly block_size tokens
    k1 = np.random.randn(num_kv_heads, block_size, head_dim).astype(np.float32)
    v1 = np.random.randn(num_kv_heads, block_size, head_dim).astype(np.float32)
    mgr.append_kv(seq_id=0, layer_idx=0, k_data=k1, v_data=v1)
    assert len(mgr.page_tables[0]) == 1
    
    # Append 1 more token -> should allocate new block
    k2 = np.random.randn(num_kv_heads, 1, head_dim).astype(np.float32)
    v2 = np.random.randn(num_kv_heads, 1, head_dim).astype(np.float32)
    mgr.append_kv(seq_id=0, layer_idx=0, k_data=k2, v_data=v2)
    assert len(mgr.page_tables[0]) == 2
    
    # Verify all data
    k_out, v_out = mgr.gather_kv(seq_id=0, layer_idx=0)
    expected_k = np.concatenate([k1, k2], axis=1)
    np.testing.assert_allclose(k_out, expected_k, atol=1e-6)
    
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Paged Attention Unit Tests")
    print("=" * 60)
    
    test_basic_allocation()
    test_append_and_gather()
    test_multi_layer()
    test_multi_sequence()
    test_out_of_blocks()
    test_gather_kv_tensor()
    test_cache_stats()
    test_reset()
    test_block_boundary_crossing()
    test_paged_vs_contiguous_attention()
    test_incremental_decode_consistency()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
