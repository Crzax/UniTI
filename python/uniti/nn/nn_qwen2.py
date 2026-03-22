"""Qwen2 model implementation for UniTi framework.
Supports DeepSeek-R1-Distill-Qwen-1.5B and similar Qwen2 architecture models.
Training-inference unified: uses UniTi ops throughout, with no_grad() to skip
autograd graph construction during inference.

Supports all backends (cpu/cuda/cpu_numpy) via UniTi's unified NDArray API.

All computation uses UniTi Tensor and ops.
All data stays on native device — no cross-backend fallbacks.
"""
from typing import Optional
from uniti.autograd import Tensor
from uniti import ops
import uniti.init as init
from .nn_basic import (
    Parameter,
    Module,
    Linear,
)
from .paged_attention import PagedKVCacheManager


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (used in Qwen2/LLaMA)."""

    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype="float32"):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape
        feature_dim = input_shape[-1]
        batch_dim = 1
        for s in input_shape[:-1]:
            batch_dim *= s

        x_flat = x.reshape((batch_dim, feature_dim))
        variance = (x_flat * x_flat).sum(axes=1).reshape((batch_dim, 1)) / feature_dim
        deno = ops.sqrt(variance + self.eps)
        deno = deno.broadcast_to((batch_dim, feature_dim))
        normed = x_flat / deno
        w = self.weight.reshape((1, feature_dim)).broadcast_to((batch_dim, feature_dim))
        out = w * normed
        return out.reshape(input_shape)


class SiLUModule(Module):
    """SiLU (Swish) activation module: x * sigmoid(x)."""

    def forward(self, x: Tensor) -> Tensor:
        return ops.silu(x)


def _precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0, device=None):
    """Precompute the cos and sin for rotary embeddings using native NDArray API.
    Returns (cos_freqs, sin_freqs) as NDArray on the specified device.
    Each has shape (max_seq_len, dim).
    """
    half_dim = dim // 2
    # Build freq indices: [0, 2, 4, ..., dim-2] / dim → exponents for theta
    # arange(half_dim) → [0, 1, ..., half_dim-1], then scale
    idx = device.arange(half_dim)  # NDArray [0, 1, ..., half_dim-1]
    idx = idx * (2.0 / dim)  # [0, 2/dim, 4/dim, ...]
    # freqs = 1 / (theta ** idx) = theta ** (-idx)
    # NDArray has __pow__ for scalar power
    freqs = idx * (-1.0)  # [-0, -2/dim, -4/dim, ...]
    import math
    log_theta = math.log(theta)
    # theta^(-idx) = exp(-idx * log(theta))
    freqs = (freqs * log_theta).exp()  # (half_dim,)
    
    # t = [0, 1, ..., max_seq_len-1]
    t = device.arange(max_seq_len)  # (max_seq_len,)
    
    # outer product: freqs_table[i,j] = t[i] * freqs[j]
    # Broadcast: t -> (max_seq_len, 1), freqs -> (1, half_dim)
    t_col = t.reshape((max_seq_len, 1)).broadcast_to((max_seq_len, half_dim))
    freqs_row = freqs.reshape((1, half_dim)).broadcast_to((max_seq_len, half_dim))
    freqs_table = t_col * freqs_row  # (max_seq_len, half_dim)
    
    # cos and sin using native NDArray methods
    cos_half = freqs_table.cos()  # (max_seq_len, half_dim)
    sin_half = freqs_table.sin()  # (max_seq_len, half_dim)
    
    # Concatenate [cos, cos] and [sin, sin] along last dim → (max_seq_len, dim)
    # NDArray doesn't have concat — use device.full + slice assignment
    cos_freqs = device.full((max_seq_len, dim), 0.0)
    cos_freqs[:, :half_dim] = cos_half
    cos_freqs[:, half_dim:dim] = cos_half
    
    sin_freqs = device.full((max_seq_len, dim), 0.0)
    sin_freqs[:, :half_dim] = sin_half
    sin_freqs[:, half_dim:dim] = sin_half
    
    return cos_freqs, sin_freqs


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply rotary position embedding — pure UniTi ops.
    x: (batch, num_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim) - will be broadcast
    """
    batch, num_heads, seq_len, head_dim = x.shape
    half = head_dim // 2

    cos_b = cos.reshape((1, 1, seq_len, head_dim)).broadcast_to(x.shape)
    sin_b = sin.reshape((1, 1, seq_len, head_dim)).broadcast_to(x.shape)

    # rotate_half(x) = [-x[..., half:], x[..., :half]]
    x1 = ops.tensor_slice(x, ((3, 0, half),))         # x[..., :half]
    x2 = ops.tensor_slice(x, ((3, half, head_dim),))   # x[..., half:]
    x_rotated = ops.concatenate([-x2, x1], axis=3)     # [-x2, x1] along last dim

    return x * cos_b + x_rotated * sin_b


def _batched_matmul(a: Tensor, b_T: Tensor) -> Tensor:
    """Batched matmul for 4D+ tensors: sum-product over last dim.
    a:   (..., M, K)
    b_T: (..., N, K)   (i.e. b transposed, so result is a @ b = a @ b_T^T -> (..., M, N))
    Implemented via reshape + broadcast + elementwise multiply + sum.
    """
    a_shape = (*a.shape[:-1], 1, a.shape[-1])
    a_exp = a.reshape(a_shape)
    b_T_shape = (*b_T.shape[:-2], 1, *b_T.shape[-2:])
    b_T_exp = b_T.reshape(b_T_shape)
    # broadcast
    bcast_a = list(a_shape)
    bcast_a[-2] = b_T.shape[-2]
    bcast_b = list(b_T_shape)
    bcast_b[-3] = a.shape[-2]
    a_exp = a_exp.broadcast_to(tuple(bcast_a))
    b_T_exp = b_T_exp.broadcast_to(tuple(bcast_b))
    return (a_exp * b_T_exp).sum(len(a_exp.shape) - 1)


class Qwen2Attention(Module):
    """Qwen2 GQA with RoPE and KV Cache for incremental decoding.
    
    Supports two KV cache modes:
      1. Contiguous cache (legacy): pre-allocated contiguous buffer per sequence.
      2. Paged cache (new): uses PagedKVCacheManager for block-based memory management.
    
    The mode is determined by which init method is called:
      - init_cache() -> contiguous mode
      - set_paged_cache() -> paged mode
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int = 131072,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.device = device
        self.dtype = dtype

        self.q_proj = Linear(hidden_size, num_attention_heads * self.head_dim, bias=True, device=device, dtype=dtype)
        self.k_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True, device=device, dtype=dtype)
        self.v_proj = Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True, device=device, dtype=dtype)
        self.o_proj = Linear(num_attention_heads * self.head_dim, hidden_size, bias=False, device=device, dtype=dtype)

        cos_freqs, sin_freqs = _precompute_freqs_cis(self.head_dim, max_position_embeddings, rope_theta, device=device)
        self._cos_cache = cos_freqs  # NDArray on device
        self._sin_cache = sin_freqs  # NDArray on device

        # --- Contiguous cache state (legacy mode) ---
        self._k_cache = None  # NDArray on device (all backends)
        self._v_cache = None  # NDArray on device (all backends)
        self._cache_len = 0  # number of valid positions in cache
        self._max_cache_len = 0  # max cache capacity

        # --- Paged cache state (new mode) ---
        self._paged_cache_mgr: Optional[PagedKVCacheManager] = None
        self._paged_layer_idx: int = -1  # which layer this attention belongs to
        self._paged_seq_id: int = 0  # current sequence id for paged mode

    def init_cache(self, batch_size: int, max_cache_len: int):
        """Pre-allocate contiguous KV cache (legacy mode).
        Uses NDArray API (device.full) to allocate directly on-device for all backends.
        """
        # Clear paged cache if switching modes
        self._paged_cache_mgr = None
        
        shape = (batch_size, self.num_kv_heads, max_cache_len, self.head_dim)
        self._max_cache_len = max_cache_len
        # Unified: allocate NDArray directly on device (cpu/cuda/cpu_numpy all support device.full)
        self._k_cache = self.device.full(shape, 0.0)
        self._v_cache = self.device.full(shape, 0.0)
        self._cache_len = 0

    def set_paged_cache(self, cache_mgr: 'PagedKVCacheManager', layer_idx: int, seq_id: int = 0):
        """Attach a paged KV cache manager to this attention layer.
        
        Args:
            cache_mgr: The shared PagedKVCacheManager instance.
            layer_idx: This layer's index (used to index into cache_mgr's per-layer blocks).
            seq_id: The sequence ID to use for cache operations.
        """
        # Clear contiguous cache if switching modes
        self._k_cache = None
        self._v_cache = None
        self._cache_len = 0
        self._max_cache_len = 0
        
        self._paged_cache_mgr = cache_mgr
        self._paged_layer_idx = layer_idx
        self._paged_seq_id = seq_id

    def reset_cache(self):
        """Reset all cache state (both contiguous and paged)."""
        self._k_cache = None
        self._v_cache = None
        self._cache_len = 0
        self._max_cache_len = 0
        self._paged_cache_mgr = None
        self._paged_layer_idx = -1

    def _softmax(self, logit: Tensor) -> Tensor:
        """Numerically stable softmax over last dimension — pure UniTi ops."""
        ndim = len(logit.shape)
        last_axis = ndim - 1
        # reduce_max returns shape with last axis kept as size 1
        max_val = ops.reduce_max(logit, last_axis)
        max_val = max_val.broadcast_to(logit.shape)
        probs = ops.exp(logit - max_val)
        denom = probs.sum(axes=last_axis)
        denom_shape = list(logit.shape)
        denom_shape[last_axis] = 1
        denom = denom.reshape(tuple(denom_shape)).broadcast_to(logit.shape)
        return probs / denom

    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """Repeat key/value heads to match query heads."""
        if n_rep == 1:
            return x
        bs, kv_heads, seq_len, head_dim = x.shape
        x = x.reshape((bs, kv_heads, 1, seq_len, head_dim))
        x = x.broadcast_to((bs, kv_heads, n_rep, seq_len, head_dim))
        x = x.reshape((bs, kv_heads * n_rep, seq_len, head_dim))
        return x

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        """
        x: (batch_size, seq_len, hidden_size)
        start_pos: position offset for KV cache (0 = no cache / training mode)
        Returns: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_length, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape: (bs, seq, heads*hd) -> (bs, heads, seq, hd)
        q = q.reshape((batch_size, seq_length, self.num_heads, self.head_dim)).transpose(axes=(1, 2))
        k = k.reshape((batch_size, seq_length, self.num_kv_heads, self.head_dim)).transpose(axes=(1, 2))
        v = v.reshape((batch_size, seq_length, self.num_kv_heads, self.head_dim)).transpose(axes=(1, 2))

        # RoPE — slice precomputed cos/sin NDArray and wrap as Tensor
        cos_nd = self._cos_cache[start_pos:start_pos + seq_length, :]
        sin_nd = self._sin_cache[start_pos:start_pos + seq_length, :]
        cos_t = Tensor.make_const(cos_nd, requires_grad=False)
        sin_t = Tensor.make_const(sin_nd, requires_grad=False)

        q = apply_rotary_emb(q, cos_t, sin_t)
        k = apply_rotary_emb(k, cos_t, sin_t)

        # KV Cache handling — choose between paged and contiguous modes
        end_pos = start_pos + seq_length
        
        if self._paged_cache_mgr is not None:
            # ====== PAGED ATTENTION MODE ======
            # Device-unified: all backends use NDArray through realize_cached_data()
            
            for b in range(batch_size):
                # Get NDArray directly from Tensor — works on all backends (cpu/cuda/cpu_numpy)
                k_nd = k.realize_cached_data()  # NDArray: (batch, num_kv_heads, seq_length, head_dim)
                v_nd = v.realize_cached_data()
                # Slice out this batch element: (1, num_kv_heads, seq_length, head_dim) -> compact -> reshape
                k_b = k_nd[b:b+1, :, :, :].compact().reshape(
                    (self.num_kv_heads, seq_length, self.head_dim))
                v_b = v_nd[b:b+1, :, :, :].compact().reshape(
                    (self.num_kv_heads, seq_length, self.head_dim))
                # append_kv accepts NDArray directly — stays on native device
                self._paged_cache_mgr.append_kv(
                    seq_id=self._paged_seq_id + b,
                    layer_idx=self._paged_layer_idx,
                    k_data=k_b,
                    v_data=v_b,
                )
            
            # Gather full KV cache from paged blocks
            if batch_size == 1:
                k, v = self._paged_cache_mgr.gather_kv_tensor(
                    seq_id=self._paged_seq_id,
                    layer_idx=self._paged_layer_idx,
                )
            else:
                # Batch mode: gather per sequence as NDArray, concatenate via NDArray, wrap as Tensor
                k_list, v_list = [], []
                for b in range(batch_size):
                    k_b, v_b = self._paged_cache_mgr.gather_kv_as_ndarray(
                        seq_id=self._paged_seq_id + b,
                        layer_idx=self._paged_layer_idx,
                    )
                    k_list.append(k_b)
                    v_list.append(v_b)
                # Concatenate NDArrays along batch dim (axis=0) using device.full + slice assignment
                total_len = k_list[0].shape[1]  # seq dim
                k_shape = (batch_size, self.num_kv_heads, total_len, self.head_dim)
                v_shape = k_shape
                k_concat = self.device.full(k_shape, 0.0)
                v_concat = self.device.full(v_shape, 0.0)
                for b in range(batch_size):
                    k_concat[b:b+1, :, :, :] = k_list[b]
                    v_concat[b:b+1, :, :, :] = v_list[b]
                k = Tensor.make_const(k_concat, requires_grad=False)
                v = Tensor.make_const(v_concat, requires_grad=False)

        elif self._k_cache is not None and self._v_cache is not None:
            # ====== CONTIGUOUS CACHE MODE (unified NDArray for all backends) ======
            k_nd = k.realize_cached_data()
            v_nd = v.realize_cached_data()
            self._k_cache[:, :, start_pos:end_pos, :] = k_nd
            self._v_cache[:, :, start_pos:end_pos, :] = v_nd
            self._cache_len = end_pos
            k = Tensor.make_const(self._k_cache[:, :, :end_pos, :], requires_grad=False)
            v = Tensor.make_const(self._v_cache[:, :, :end_pos, :], requires_grad=False)

        total_len = k.shape[2]

        # GQA: repeat k, v
        k = self._repeat_kv(k, self.num_kv_groups)
        v = self._repeat_kv(v, self.num_kv_groups)

        # Attention: Q @ K^T / sqrt(d) — scale as float scalar
        scale = float(self.head_dim ** 0.5)
        attn_weights = _batched_matmul(q, k) / scale

        # Causal mask — built using native NDArray triu_mask
        if seq_length > 1:
            triu_k = total_len - seq_length + 1
            mask_nd = self.device.triu_mask(seq_length, total_len, k=triu_k)
            mask_t = Tensor.make_const(mask_nd, requires_grad=False)
            mask_t = mask_t.reshape((1, 1, seq_length, total_len)).broadcast_to(attn_weights.shape)
            attn_weights = attn_weights + mask_t

        attn_weights = self._softmax(attn_weights)

        # attn @ v
        v_T = v.transpose(axes=(2, 3))
        attn_output = _batched_matmul(attn_weights, v_T)

        # (bs, heads, seq, hd) -> (bs, seq, hidden)
        attn_output = attn_output.transpose(axes=(1, 2))
        attn_output = attn_output.reshape((batch_size, seq_length, self.num_heads * self.head_dim))

        return self.o_proj(attn_output)


class Qwen2MLP(Module):
    """Qwen2 MLP with SwiGLU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int, device=None, dtype="float32"):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        gate = ops.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Qwen2DecoderLayer(Module):
    """A single Qwen2 transformer decoder layer."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 131072,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.self_attn = Qwen2Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            device=device,
            dtype=dtype,
        )
        self.mlp = Qwen2MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
            dtype=dtype,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        # Pre-norm attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, start_pos=start_pos)
        x = residual + x

        # Pre-norm MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class QwenEmbedding(Module):
    """Embedding layer using native NDArray embedding_lookup.

    All operations stay on the native device — no numpy transfers needed.
    The weight matrix is stored as a Tensor (NDArray backed), and embedding
    lookup is performed directly on-device via the embedding_lookup kernel.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype="float32"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, mean=0, std=1, device=device, dtype=dtype)
        )

    def _invalidate_cache(self):
        """No-op: kept for backward compatibility. Native embedding needs no cache."""
        pass

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, seq_len) - float tensor of token IDs
        Returns: (batch_size, seq_len, embedding_dim)

        Uses native device.embedding_lookup kernel — all on-device, no numpy.
        """
        input_shape = x.shape
        # Flatten IDs to 1D NDArray
        ids_nd = x.realize_cached_data().compact().reshape((x.shape[0] * x.shape[1],))
        # Get weight as NDArray (compact flat)
        weight_nd = self.weight.realize_cached_data().compact().reshape(
            (self.num_embeddings * self.embedding_dim,))
        # Actually we need weight as 2D for the kernel — pass flat + embedding_dim
        num_ids = ids_nd.shape[0]
        result_nd = self.device.embedding_lookup(weight_nd, ids_nd, self.embedding_dim)
        # Reshape to (batch_size, seq_len, embedding_dim)
        result_nd = result_nd.reshape((*input_shape, self.embedding_dim))
        return Tensor.make_const(result_nd, requires_grad=False)


class Qwen2Model(Module):
    """The Qwen2 base model (without LM head)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 131072,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.device = device
        self.dtype = dtype
        
        self.embed_tokens = QwenEmbedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.layers = [
            Qwen2DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                attention_dropout=attention_dropout,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps, device=device, dtype=dtype)
        
        # Paged cache manager reference
        self._paged_cache_mgr: Optional[PagedKVCacheManager] = None

    def init_cache(self, batch_size: int, max_cache_len: int):
        """Initialize contiguous KV cache (legacy mode)."""
        for layer in self.layers:
            layer.self_attn.init_cache(batch_size, max_cache_len)

    def init_paged_cache(
        self,
        block_size: int = 16,
        max_num_blocks: int = 256,
        seq_id: int = 0,
        initial_len: int = 0,
    ) -> PagedKVCacheManager:
        """Initialize paged KV cache for all attention layers.
        
        Creates a shared PagedKVCacheManager and attaches it to all layers.
        
        Args:
            block_size: Number of tokens per page/block.
            max_num_blocks: Maximum number of physical blocks in the pool.
            seq_id: Sequence ID to allocate.
            initial_len: Pre-allocate blocks for this many tokens.
            
        Returns:
            The created PagedKVCacheManager for external inspection/control.
        """
        cache_mgr = PagedKVCacheManager(
            num_layers=self.num_hidden_layers,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Allocate sequence
        cache_mgr.allocate_sequence(seq_id, initial_len=initial_len)
        
        # Attach to all layers
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.set_paged_cache(cache_mgr, layer_idx, seq_id)
        
        self._paged_cache_mgr = cache_mgr
        return cache_mgr

    def reset_cache(self):
        """Reset all caches (both contiguous and paged)."""
        for layer in self.layers:
            layer.self_attn.reset_cache()
        if self._paged_cache_mgr is not None:
            self._paged_cache_mgr.reset()
            self._paged_cache_mgr = None

    def forward(self, input_ids_tensor: Tensor, start_pos: int = 0) -> Tensor:
        """
        input_ids_tensor: (batch_size, seq_len) float tensor of token IDs
        start_pos: position offset for KV cache
        Returns: (batch_size, seq_len, hidden_size)
        """
        x = self.embed_tokens(input_ids_tensor)
        for layer in self.layers:
            x = layer(x, start_pos=start_pos)
        x = self.norm(x)
        return x


class Qwen2ForCausalLM(Module):
    """Qwen2 model with a language modeling head.
    
    Supports two KV cache modes for inference:
      1. Contiguous: model.init_cache(batch_size, max_cache_len)
      2. Paged: model.init_paged_cache(block_size=16, max_num_blocks=256)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 131072,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.model = Qwen2Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            device=device,
            dtype=dtype,
        )
        self.lm_head = Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype

    def init_cache(self, batch_size: int, max_cache_len: int):
        """Initialize contiguous KV cache (legacy mode)."""
        self.model.init_cache(batch_size, max_cache_len)

    def init_paged_cache(
        self,
        block_size: int = 16,
        max_num_blocks: int = 256,
        seq_id: int = 0,
        initial_len: int = 0,
    ) -> 'PagedKVCacheManager':
        """Initialize paged KV cache for all attention layers.
        
        This is the recommended mode for inference — it allocates memory
        on-demand in fixed-size blocks, avoiding the need to pre-allocate
        max_seq_len of contiguous memory per sequence.
        
        Args:
            block_size: Number of tokens per page/block (default: 16).
            max_num_blocks: Maximum number of physical blocks in the pool.
            seq_id: Sequence ID to allocate.
            initial_len: Pre-allocate blocks for this many tokens.
            
        Returns:
            The created PagedKVCacheManager for monitoring/stats.
        """
        return self.model.init_paged_cache(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            seq_id=seq_id,
            initial_len=initial_len,
        )

    def reset_cache(self):
        """Reset all caches."""
        self.model.reset_cache()

    def forward(self, input_ids_tensor: Tensor, start_pos: int = 0, last_only: bool = False) -> Tensor:
        """
        input_ids_tensor: (batch_size, seq_len)
        start_pos: position offset for KV cache
        last_only: if True, only compute lm_head for the last token (decode optimization)
        Returns: logits (batch_size, seq_len or 1, vocab_size)
        """
        hidden_states = self.model(input_ids_tensor, start_pos=start_pos)
        if last_only:
            # Only compute logits for the last token — use UniTi slice
            seq_len = hidden_states.shape[1]
            last_hidden = hidden_states[:, seq_len - 1:seq_len, :]
            logits = self.lm_head(last_hidden)
        else:
            logits = self.lm_head(hidden_states)
        return logits
