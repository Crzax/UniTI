"""Paged Attention for UniTi framework.

Implements vLLM-style Paged KV Cache for efficient LLM inference:
  - Physical memory is divided into fixed-size blocks (pages).
  - Each sequence maintains a logical->physical block mapping (page table).
  - KV cache is stored in pre-allocated physical block pools.
  - Memory is allocated on-demand and can be freed/reused across sequences.

This avoids the contiguous pre-allocation problem where each sequence must
reserve max_seq_len of KV cache memory upfront.

Compatible with UniTi's training-inference unified architecture:
  - Three backends: cpu (custom C++), cpu_numpy (numpy wrapper), cuda (custom GPU).
  - All backends use the unified NDArray API: device.full(), device.empty(),
    NDArray slice assignment (__setitem__), .compact(), .numpy(), etc.
  - No backend-specific branching needed — same code path for all devices.
  - No cross-device data transfers during append/gather — zero-copy on-device operations.

References:
  - vLLM: Efficient Memory Management for Large Language Model Serving
    with PagedAttention (Kwon et al., 2023)
"""
from typing import Any, Optional, List, Dict, Tuple, Union
from uniti.autograd import Tensor
from uniti import ops
from .nn_basic import Module
from uniti.backend_selection import NDArray, array_api


class PagedKVCacheManager:
    """Manages paged KV cache physical memory blocks and page tables.
    
    Memory layout:
      - Physical KV blocks: (num_blocks, num_kv_heads, block_size, head_dim)
      - Page table per sequence: list of physical block indices
      
    Each "page" (block) stores `block_size` consecutive KV pairs.
    Sequences map logical block indices to physical block indices via page tables.
    
    Device-unified design:
      All three backends (cpu, cuda, cpu_numpy) use the same NDArray API:
      - device.full(shape, val) for allocation
      - NDArray slice assignment for writing KV data
      - .compact() for making sliced views contiguous
      - .numpy() for converting to numpy when needed
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_num_blocks: int = 256,
        device=None,
        dtype: str = "float32",
    ):
        """
        Args:
            num_layers: Number of transformer layers (each layer has its own KV cache).
            num_kv_heads: Number of key/value heads (for GQA).
            head_dim: Dimension per attention head.
            block_size: Number of tokens per block (page size).
            max_num_blocks: Maximum number of physical blocks in the pool.
            device: UniTi device (cpu/cuda/cpu_numpy).
            dtype: Data type for cache storage.
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.device = device
        self.dtype = dtype

        # Physical block pool: per-layer K and V caches
        # Shape per layer: (max_num_blocks, num_kv_heads, block_size, head_dim)
        # All backends use device.full() to allocate NDArray directly on device
        block_shape = (max_num_blocks, num_kv_heads, block_size, head_dim)
        
        self.k_cache_blocks = [
            self.device.full(block_shape, 0.0)
            for _ in range(num_layers)
        ]
        self.v_cache_blocks: list[Any] = [
            self.device.full(block_shape, 0.0)
            for _ in range(num_layers)
        ]

        # Free block list: all blocks start as free
        self.free_blocks: List[int] = list(range(max_num_blocks))
        
        # Page tables: mapping from sequence_id -> list of physical block indices
        # Each sequence has a list of block indices, where logical_block[i] -> physical_block[page_table[i]]
        self.page_tables: Dict[int, List[int]] = {}
        # Track the number of filled tokens per (seq_id, layer_idx)
        # This allows each layer to append independently (same tokens go through all layers)
        self.seq_layer_lengths: Dict[int, Dict[int, int]] = {}  # seq_id -> {layer_idx -> length}

    @property
    def num_free_blocks(self) -> int:
        """Number of available physical blocks."""
        return len(self.free_blocks)
    
    @property
    def num_used_blocks(self) -> int:
        """Number of blocks currently in use."""
        return self.max_num_blocks - len(self.free_blocks)

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if we can allocate the requested number of blocks."""
        return len(self.free_blocks) >= num_blocks

    def _allocate_block(self) -> int:
        """Allocate a single physical block. Returns block index."""
        if not self.free_blocks:
            raise RuntimeError(
                f"PagedKVCache: Out of physical blocks! "
                f"max_num_blocks={self.max_num_blocks}, all in use. "
                f"Consider increasing max_num_blocks or reducing batch/sequence size."
            )
        return self.free_blocks.pop(0)

    def _free_block(self, block_idx: int):
        """Return a physical block to the free pool."""
        if block_idx not in self.free_blocks:
            self.free_blocks.append(block_idx)

    def allocate_sequence(self, seq_id: int, initial_len: int = 0):
        """Register a new sequence and allocate initial blocks.
        
        Args:
            seq_id: Unique identifier for the sequence.
            initial_len: Number of tokens to pre-allocate blocks for (e.g., prompt length).
        """
        if seq_id in self.page_tables:
            raise ValueError(f"Sequence {seq_id} already allocated.")
        
        num_blocks_needed = (initial_len + self.block_size - 1) // self.block_size if initial_len > 0 else 0
        
        if not self.can_allocate(num_blocks_needed):
            raise RuntimeError(
                f"Cannot allocate {num_blocks_needed} blocks for sequence {seq_id}. "
                f"Only {self.num_free_blocks} blocks available."
            )
        
        blocks = [self._allocate_block() for _ in range(num_blocks_needed)]
        self.page_tables[seq_id] = blocks
        self.seq_layer_lengths[seq_id] = {i: initial_len for i in range(self.num_layers)}

    def free_sequence(self, seq_id: int):
        """Free all blocks belonging to a sequence."""
        if seq_id not in self.page_tables:
            return
        for block_idx in self.page_tables[seq_id]:
            self._free_block(block_idx)
        del self.page_tables[seq_id]
        if seq_id in self.seq_layer_lengths:
            del self.seq_layer_lengths[seq_id]

    @property
    def seq_lengths(self) -> Dict[int, int]:
        """Return effective sequence lengths (max across layers per sequence).
        Provided for backward compatibility and stats reporting.
        """
        result = {}
        for seq_id, layer_dict in self.seq_layer_lengths.items():
            result[seq_id] = max(layer_dict.values()) if layer_dict else 0
        return result

    def _ensure_blocks(self, seq_id: int, target_len: int):
        """Ensure enough blocks are allocated for target_len tokens."""
        num_blocks_needed = (target_len + self.block_size - 1) // self.block_size
        current_blocks = len(self.page_tables[seq_id])
        
        while current_blocks < num_blocks_needed:
            new_block = self._allocate_block()
            self.page_tables[seq_id].append(new_block)
            current_blocks += 1

    def append_kv(
        self,
        seq_id: int,
        layer_idx: int,
        k_data,
        v_data,
    ):
        """Append new KV pairs to a sequence's cache for a specific layer.
        
        Device-unified: accepts NDArray on any device (cpu/cuda/cpu_numpy).
        Uses NDArray slice assignment — no cross-device transfer, all on-device.
        
        Args:
            seq_id: Sequence identifier.
            layer_idx: Transformer layer index.
            k_data: Key data, shape (num_kv_heads, new_tokens, head_dim) as NDArray.
            v_data: Value data, shape (num_kv_heads, new_tokens, head_dim) as NDArray.
        """
        if seq_id not in self.page_tables:
            raise ValueError(f"Sequence {seq_id} not allocated.")
        
        cur_len = self.seq_layer_lengths[seq_id].get(layer_idx, 0)
        new_tokens = k_data.shape[1]
        target_len = cur_len + new_tokens
        
        # Ensure we have enough blocks
        self._ensure_blocks(seq_id, target_len)
        
        page_table = self.page_tables[seq_id]
        k_blocks = self.k_cache_blocks[layer_idx]
        v_blocks = self.v_cache_blocks[layer_idx]
        
        # Write token by token using NDArray slice assignment (all on-device)
        for i in range(new_tokens):
            pos = cur_len + i
            logical_block = pos // self.block_size
            offset_in_block = pos % self.block_size
            physical_block = page_table[logical_block]
            
            # Slice the token from input: (num_kv_heads, 1, head_dim) -> compact -> reshape
            k_token = k_data[:, i:i+1, :].compact().reshape(
                (1, self.num_kv_heads, 1, self.head_dim))
            v_token = v_data[:, i:i+1, :].compact().reshape(
                (1, self.num_kv_heads, 1, self.head_dim))
            
            # Write directly on-device via NDArray __setitem__ (ewise_setitem)
            k_blocks[physical_block:physical_block+1, :, offset_in_block:offset_in_block+1, :] = k_token
            v_blocks[physical_block:physical_block+1, :, offset_in_block:offset_in_block+1, :] = v_token
        
        self.seq_layer_lengths[seq_id][layer_idx] = target_len

    def gather_kv(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple:
        """Gather the full KV cache for a sequence as numpy arrays.
        
        Reads from paged NDArray blocks, converts to numpy for output.
        
        Args:
            seq_id: Sequence identifier.
            layer_idx: Transformer layer index.
            
        Returns:
            (k_cache, v_cache): Each of shape (num_kv_heads, seq_len, head_dim) as numpy.
        """
        if seq_id not in self.page_tables:
            raise ValueError(f"Sequence {seq_id} not allocated.")
        
        seq_len = self.seq_layer_lengths[seq_id].get(layer_idx, 0)
        if seq_len == 0:
            shape = (self.num_kv_heads, 0, self.head_dim)
            # Allocate empty on device, then convert to numpy
            k_empty = self.device.full(shape, 0.0)
            v_empty = self.device.full(shape, 0.0)
            return k_empty.numpy(), v_empty.numpy()
        
        page_table = self.page_tables[seq_id]
        k_blocks = self.k_cache_blocks[layer_idx]
        v_blocks = self.v_cache_blocks[layer_idx]
        
        # Allocate output NDArray on device, then gather block by block
        out_shape = (self.num_kv_heads, seq_len, self.head_dim)
        k_out = self.device.full(out_shape, 0.0)
        v_out = self.device.full(out_shape, 0.0)
        
        # Copy block by block using NDArray slice ops
        for logical_idx, physical_idx in enumerate(page_table):
            start_in_seq = logical_idx * self.block_size
            tokens_in_block = min(self.block_size, seq_len - start_in_seq)
            if tokens_in_block <= 0:
                break
            
            # Source: (1, num_kv_heads, tokens_in_block, head_dim) -> reshape to (num_kv_heads, tokens_in_block, head_dim)
            k_src = k_blocks[physical_idx:physical_idx+1, :, :tokens_in_block, :].compact().reshape(
                (self.num_kv_heads, tokens_in_block, self.head_dim))
            v_src = v_blocks[physical_idx:physical_idx+1, :, :tokens_in_block, :].compact().reshape(
                (self.num_kv_heads, tokens_in_block, self.head_dim))
            
            # Write to output via NDArray slice assignment
            k_out[:, start_in_seq:start_in_seq+tokens_in_block, :] = k_src
            v_out[:, start_in_seq:start_in_seq+tokens_in_block, :] = v_src
        
        # Convert final result to numpy
        return k_out.numpy(), v_out.numpy()

    def gather_kv_as_ndarray(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[NDArray, NDArray]:
        """Gather the full KV cache directly as NDArray on the native device.
        
        All operations stay on-device (no numpy transfer).
        Output shape: (1, num_kv_heads, seq_len, head_dim).
        
        Note: Each call allocates a new output NDArray of shape (1, num_kv_heads, seq_len, head_dim).
        This is intentional — seq_len grows each step, so we can't reuse a fixed buffer.
        But allocation is done directly on-device via device.full(), NOT via numpy transfer.
        
        Returns:
            (k_cache, v_cache): Each of shape (1, num_kv_heads, seq_len, head_dim) as NDArray.
        """
        if seq_id not in self.page_tables:
            raise ValueError(f"Sequence {seq_id} not allocated.")
        
        seq_len = self.seq_layer_lengths[seq_id].get(layer_idx, 0)
        if seq_len == 0:
            shape = (1, self.num_kv_heads, 0, self.head_dim)
            return self.device.full(shape, 0.0), self.device.full(shape, 0.0)
        
        page_table = self.page_tables[seq_id]
        k_blocks = self.k_cache_blocks[layer_idx]
        v_blocks = self.v_cache_blocks[layer_idx]
        
        # Allocate output NDArray DIRECTLY on-device (device.full -> NDArray.make -> device.Array -> fill)
        out_shape = (1, self.num_kv_heads, seq_len, self.head_dim)
        k_out = self.device.full(out_shape, 0.0)
        v_out = self.device.full(out_shape, 0.0)
        
        # Copy block by block — all operations stay on the native device
        for logical_idx, physical_idx in enumerate(page_table):
            start_in_seq = logical_idx * self.block_size
            tokens_in_block = min(self.block_size, seq_len - start_in_seq)
            if tokens_in_block <= 0:
                break
            
            # Source: slice from cache block, compact to make contiguous
            k_src = k_blocks[physical_idx:physical_idx+1, :, :tokens_in_block, :].compact()
            v_src = v_blocks[physical_idx:physical_idx+1, :, :tokens_in_block, :].compact()
            
            # Destination slice in output — NDArray __setitem__ handles on-device copy
            k_out[:, :, start_in_seq:start_in_seq+tokens_in_block, :] = k_src
            v_out[:, :, start_in_seq:start_in_seq+tokens_in_block, :] = v_src
        
        return k_out, v_out

    def gather_kv_tensor(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        """Gather KV cache and return as UniTi Tensors.
        
        Device-unified: gathers directly on-device via NDArray ops, wraps as Tensor.
        No numpy intermediate — uses Tensor.make_const(NDArray) directly.
        
        Returns:
            (k_tensor, v_tensor): Each of shape (1, num_kv_heads, seq_len, head_dim)
        """
        k_nd, v_nd = self.gather_kv_as_ndarray(seq_id, layer_idx)
        k_tensor = Tensor.make_const(k_nd, requires_grad=False)
        v_tensor = Tensor.make_const(v_nd, requires_grad=False)
        return k_tensor, v_tensor

    def reset(self):
        """Reset all allocations — free all sequences and blocks."""
        # Free all sequences
        for seq_id in list(self.page_tables.keys()):
            self.free_sequence(seq_id)
        # Reset free list
        self.free_blocks = list(range(self.max_num_blocks))
        self.seq_layer_lengths.clear()
        
        # Re-allocate zeroed block memory directly on device
        block_shape = (self.max_num_blocks, self.num_kv_heads, self.block_size, self.head_dim)
        for layer_idx in range(self.num_layers):
            self.k_cache_blocks[layer_idx] = self.device.full(block_shape, 0.0)
            self.v_cache_blocks[layer_idx] = self.device.full(block_shape, 0.0)

    def get_cache_stats(self) -> dict:
        """Return cache utilization statistics."""
        seq_lens = self.seq_lengths
        return {
            "num_layers": self.num_layers,
            "block_size": self.block_size,
            "max_num_blocks": self.max_num_blocks,
            "num_free_blocks": self.num_free_blocks,
            "num_used_blocks": self.num_used_blocks,
            "utilization": self.num_used_blocks / self.max_num_blocks * 100,
            "num_sequences": len(self.page_tables),
            "sequences": {
                seq_id: {
                    "length": seq_lens.get(seq_id, 0),
                    "num_blocks": len(self.page_tables[seq_id]),
                    "block_ids": self.page_tables[seq_id],
                }
                for seq_id in self.page_tables
            },
        }
