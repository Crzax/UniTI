"""Paged Attention for UniTi framework.

Implements vLLM-style Paged KV Cache for efficient LLM inference:
  - Physical memory is divided into fixed-size blocks (pages).
  - Each sequence maintains a logical->physical block mapping (page table).
  - KV cache is stored in pre-allocated physical block pools.
  - Memory is allocated on-demand and can be freed/reused across sequences.

This avoids the contiguous pre-allocation problem where each sequence must
reserve max_seq_len of KV cache memory upfront.

Compatible with UniTi's training-inference unified architecture:
  - All data stays on the native device: GPU data never crosses to CPU, CPU data stays on CPU.
  - Uses NDArray operations for GPU mode, numpy operations for CPU mode.
  - No CPU↔GPU data transfers during append/gather — zero-copy on-device operations.

References:
  - vLLM: Efficient Memory Management for Large Language Model Serving
    with PagedAttention (Kwon et al., 2023)
"""
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
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
      - GPU mode: blocks are NDArray on cuda, append/gather use NDArray slice ops (zero CPU transfer)
      - CPU mode: blocks are numpy arrays, append/gather use numpy indexing
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

        # Detect device type
        self._is_cuda = (device is not None and 
                         hasattr(device, 'name') and 
                         device.name == 'cuda')
        self._is_cpu_ndarray = (device is not None and 
                                hasattr(device, 'name') and 
                                device.name in ('cpu', 'cpu_numpy'))

        # Physical block pool: per-layer K and V caches
        # Shape per layer: (max_num_blocks, num_kv_heads, block_size, head_dim)
        block_shape = (max_num_blocks, num_kv_heads, block_size, head_dim)
        
        if self._is_cuda:
            # GPU mode: allocate NDArray blocks on device
            self.k_cache_blocks = [
                array_api.NDArray(np.zeros(block_shape, dtype=np.float32), device=device)
                for _ in range(num_layers)
            ]
            self.v_cache_blocks = [
                array_api.NDArray(np.zeros(block_shape, dtype=np.float32), device=device)
                for _ in range(num_layers)
            ]
        else:
            # CPU mode: numpy arrays
            self.k_cache_blocks = [
                np.zeros(block_shape, dtype=np.float32)
                for _ in range(num_layers)
            ]
            self.v_cache_blocks = [
                np.zeros(block_shape, dtype=np.float32)
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
        
        Device-unified: accepts NDArray (GPU) or numpy (CPU) directly.
        No cross-device data transfer occurs — data stays on its native device.
        
        Args:
            seq_id: Sequence identifier.
            layer_idx: Transformer layer index.
            k_data: Key data, shape (num_kv_heads, new_tokens, head_dim).
                    NDArray for GPU mode, numpy array for CPU mode.
            v_data: Value data, shape (num_kv_heads, new_tokens, head_dim).
                    NDArray for GPU mode, numpy array for CPU mode.
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
        
        if self._is_cuda:
            # GPU mode: k_data/v_data are NDArray on cuda
            # Write token by token using NDArray slice assignment (all on GPU, no CPU transfer)
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
                
                # Write directly on GPU via NDArray ewise_setitem
                k_blocks[physical_block:physical_block+1, :, offset_in_block:offset_in_block+1, :] = k_token
                v_blocks[physical_block:physical_block+1, :, offset_in_block:offset_in_block+1, :] = v_token
        else:
            # CPU mode: k_data/v_data are numpy arrays
            for i in range(new_tokens):
                pos = cur_len + i
                logical_block = pos // self.block_size
                offset_in_block = pos % self.block_size
                physical_block = page_table[logical_block]
                k_blocks[physical_block, :, offset_in_block, :] = k_data[:, i, :]
                v_blocks[physical_block, :, offset_in_block, :] = v_data[:, i, :]
        
        self.seq_layer_lengths[seq_id][layer_idx] = target_len

    def gather_kv(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gather the full KV cache for a sequence by reading from paged blocks.
        CPU-only path: returns numpy arrays.
        
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
            return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
        
        page_table = self.page_tables[seq_id]
        k_blocks = self.k_cache_blocks[layer_idx]
        v_blocks = self.v_cache_blocks[layer_idx]
        
        # Pre-allocate output
        k_out = np.zeros((self.num_kv_heads, seq_len, self.head_dim), dtype=np.float32)
        v_out = np.zeros((self.num_kv_heads, seq_len, self.head_dim), dtype=np.float32)
        
        # Copy block by block
        for logical_idx, physical_idx in enumerate(page_table):
            start_in_seq = logical_idx * self.block_size
            tokens_in_block = min(self.block_size, seq_len - start_in_seq)
            if tokens_in_block <= 0:
                break
            
            if self._is_cuda:
                # GPU -> numpy (only used when explicitly requesting numpy output)
                k_slice = k_blocks[physical_idx:physical_idx+1, :, :tokens_in_block, :]
                v_slice = v_blocks[physical_idx:physical_idx+1, :, :tokens_in_block, :]
                k_block_data = k_slice.numpy().reshape(
                    self.num_kv_heads, tokens_in_block, self.head_dim)
                v_block_data = v_slice.numpy().reshape(
                    self.num_kv_heads, tokens_in_block, self.head_dim)
                k_out[:, start_in_seq:start_in_seq+tokens_in_block, :] = k_block_data
                v_out[:, start_in_seq:start_in_seq+tokens_in_block, :] = v_block_data
            else:
                k_out[:, start_in_seq:start_in_seq+tokens_in_block, :] = \
                    k_blocks[physical_idx, :, :tokens_in_block, :]
                v_out[:, start_in_seq:start_in_seq+tokens_in_block, :] = \
                    v_blocks[physical_idx, :, :tokens_in_block, :]
        
        return k_out, v_out

    def gather_kv_as_ndarray(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[NDArray, NDArray]:
        """Gather the full KV cache directly as NDArray on the native device.
        
        GPU mode: output stays on GPU (no CPU transfer!).
        CPU mode: output is NDArray on CPU device.
        
        Note: Each call allocates a new output NDArray of shape (1, num_kv_heads, seq_len, head_dim).
        This is intentional — seq_len grows each step, so we can't reuse a fixed buffer.
        But allocation is done directly on-device (device.full), NOT via numpy→GPU transfer.
        
        Returns:
            (k_cache, v_cache): Each of shape (1, num_kv_heads, seq_len, head_dim) as NDArray.
        """
        if seq_id not in self.page_tables:
            raise ValueError(f"Sequence {seq_id} not allocated.")
        
        seq_len = self.seq_layer_lengths[seq_id].get(layer_idx, 0)
        if seq_len == 0:
            shape = (1, self.num_kv_heads, 0, self.head_dim)
            # Even for empty, allocate on-device directly
            return (self.device.full(shape, 0.0) if self._is_cuda 
                    else array_api.NDArray(np.zeros(shape, dtype=np.float32), device=self.device),
                    self.device.full(shape, 0.0) if self._is_cuda 
                    else array_api.NDArray(np.zeros(shape, dtype=np.float32), device=self.device))
        
        page_table = self.page_tables[seq_id]
        k_blocks = self.k_cache_blocks[layer_idx]
        v_blocks = self.v_cache_blocks[layer_idx]
        
        # Allocate output NDArray DIRECTLY on-device (no numpy → GPU transfer!)
        # device.full() calls NDArray.make() which allocates device memory directly,
        # then fills with zeros via device.fill() — all on GPU, no CPU involvement.
        out_shape = (1, self.num_kv_heads, seq_len, self.head_dim)
        k_out = self.device.full(out_shape, 0.0)
        v_out = self.device.full(out_shape, 0.0)
        
        # Copy block by block — all operations stay on the native device
        for logical_idx, physical_idx in enumerate(page_table):
            start_in_seq = logical_idx * self.block_size
            tokens_in_block = min(self.block_size, seq_len - start_in_seq)
            if tokens_in_block <= 0:
                break
            
            # Source: (1, num_kv_heads, tokens_in_block, head_dim) from cache block
            k_src = k_blocks[physical_idx:physical_idx+1, :, :tokens_in_block, :]
            v_src = v_blocks[physical_idx:physical_idx+1, :, :tokens_in_block, :]
            
            # Compact before assignment (slice may not be compact)
            k_src_c = k_src.compact()
            v_src_c = v_src.compact()
            
            # Destination slice in output
            k_out[:, :, start_in_seq:start_in_seq+tokens_in_block, :] = k_src_c
            v_out[:, :, start_in_seq:start_in_seq+tokens_in_block, :] = v_src_c
        
        return k_out, v_out

    def gather_kv_tensor(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        """Gather KV cache and return as UniTi Tensors.
        
        Device-unified: 
          - GPU mode: gathers directly on GPU via NDArray ops, wraps as Tensor (zero CPU transfer).
          - CPU mode: gathers as numpy, wraps as Tensor.
        
        Returns:
            (k_tensor, v_tensor): Each of shape (1, num_kv_heads, seq_len, head_dim)
        """
        if self._is_cuda:
            # GPU path: gather entirely on GPU, wrap NDArray as Tensor directly
            k_nd, v_nd = self.gather_kv_as_ndarray(seq_id, layer_idx)
            k_tensor = Tensor.make_const(k_nd, requires_grad=False)
            v_tensor = Tensor.make_const(v_nd, requires_grad=False)
            return k_tensor, v_tensor
        else:
            # CPU path: gather as numpy, construct Tensor
            k_np, v_np = self.gather_kv(seq_id, layer_idx)
            k_np = k_np[np.newaxis, :, :, :]
            v_np = v_np[np.newaxis, :, :, :]
            k_tensor = Tensor(k_np, device=self.device, dtype=self.dtype, requires_grad=False)
            v_tensor = Tensor(v_np, device=self.device, dtype=self.dtype, requires_grad=False)
            return k_tensor, v_tensor

    def reset(self):
        """Reset all allocations — free all sequences and blocks."""
        # Free all sequences
        for seq_id in list(self.page_tables.keys()):
            self.free_sequence(seq_id)
        # Reset free list
        self.free_blocks = list(range(self.max_num_blocks))
        self.seq_layer_lengths.clear()
        
        # Zero out block memory
        block_shape = (self.max_num_blocks, self.num_kv_heads, self.block_size, self.head_dim)
        for layer_idx in range(self.num_layers):
            if self._is_cuda:
                self.k_cache_blocks[layer_idx] = array_api.NDArray(
                    np.zeros(block_shape, dtype=np.float32), device=self.device)
                self.v_cache_blocks[layer_idx] = array_api.NDArray(
                    np.zeros(block_shape, dtype=np.float32), device=self.device)
            else:
                self.k_cache_blocks[layer_idx][:] = 0
                self.v_cache_blocks[layer_idx][:] = 0

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
