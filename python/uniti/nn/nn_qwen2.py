"""Qwen2 model implementation for UniTi framework.
Supports DeepSeek-R1-Distill-Qwen-1.5B and similar Qwen2 architecture models.
Training-inference unified: uses UniTi ops throughout, with no_grad() to skip
autograd graph construction during inference.

Supports both CPU and GPU backends via UniTi's device abstraction.

All computation uses UniTi Tensor and ops — numpy is only used for:
  1. _precompute_freqs_cis (one-time precomputation of RoPE constants)
  2. KV cache buffer storage (mutable state, not part of computation graph)
"""
from typing import Optional
from uniti.autograd import Tensor
from uniti import ops
import uniti.init as init
import numpy as np
from .nn_basic import (
    Parameter,
    Module,
    Linear,
)
from uniti.backend_selection import NDArray, array_api


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


def _precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute the cos and sin for rotary embeddings.
    This is a one-time precomputation — uses numpy to build constant arrays
    that are later sliced and converted to Tensor on each forward pass.
    """
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)  # (max_seq_len, dim//2)
    cos_freqs = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1)
    sin_freqs = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1)
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
    """Qwen2 GQA with RoPE and pre-allocated KV Cache for incremental decoding."""

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

        cos_freqs, sin_freqs = _precompute_freqs_cis(self.head_dim, max_position_embeddings, rope_theta)
        self._cos_cache = cos_freqs  # numpy
        self._sin_cache = sin_freqs  # numpy

        # Pre-allocated KV cache: (B, kv_heads, max_cache_len, head_dim)
        # Stored as NDArray on the same device as model for GPU support
        self._k_cache = None  # NDArray on device (GPU mode)
        self._v_cache = None  # NDArray on device (GPU mode)
        self._cache_np_k = None  # numpy buffer (CPU mode)
        self._cache_np_v = None  # numpy buffer (CPU mode)
        self._cache_len = 0  # number of valid positions in cache
        self._max_cache_len = 0  # max cache capacity

    def init_cache(self, batch_size: int, max_cache_len: int):
        """Pre-allocate KV cache arrays to avoid per-step concatenation.
        For GPU: allocates NDArray directly on device to avoid CPU<->GPU transfers.
        For CPU: uses numpy arrays.
        """
        shape = (batch_size, self.num_kv_heads, max_cache_len, self.head_dim)
        self._max_cache_len = max_cache_len
        # Check if using GPU (CUDA device)
        is_cuda = (self.device is not None and 
                   hasattr(self.device, 'name') and 
                   self.device.name == 'cuda')
        if is_cuda:
            # GPU mode: allocate directly on device, no numpy buffers needed
            self._k_cache = array_api.NDArray(np.zeros(shape, dtype=np.float32), device=self.device)
            self._v_cache = array_api.NDArray(np.zeros(shape, dtype=np.float32), device=self.device)
            self._cache_np_k = None
            self._cache_np_v = None
        else:
            # CPU mode: use numpy buffers
            self._cache_np_k = np.zeros(shape, dtype=np.float32)
            self._cache_np_v = np.zeros(shape, dtype=np.float32)
            self._k_cache = None
            self._v_cache = None
        self._cache_len = 0

    def reset_cache(self):
        self._k_cache = None
        self._v_cache = None
        self._cache_np_k = None
        self._cache_np_v = None
        self._cache_len = 0
        self._max_cache_len = 0

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

        # RoPE — slice precomputed cos/sin constants and wrap as Tensor
        cos_t = Tensor(
            self._cos_cache[start_pos:start_pos + seq_length],
            device=self.device, dtype=self.dtype, requires_grad=False
        )
        sin_t = Tensor(
            self._sin_cache[start_pos:start_pos + seq_length],
            device=self.device, dtype=self.dtype, requires_grad=False
        )

        q = apply_rotary_emb(q, cos_t, sin_t)
        k = apply_rotary_emb(k, cos_t, sin_t)

        # KV Cache handling
        end_pos = start_pos + seq_length
        
        if self._k_cache is not None and self._v_cache is not None:
            # GPU mode: update cache directly on device (no CPU<->GPU transfer!)
            # Get the underlying NDArray from Tensor's cached_data
            k_nd = k.realize_cached_data()
            v_nd = v.realize_cached_data()
            # Direct setitem on GPU NDArray
            self._k_cache[:, :, start_pos:end_pos, :] = k_nd
            self._v_cache[:, :, start_pos:end_pos, :] = v_nd
            self._cache_len = end_pos
            # Wrap sliced cache as Tensor using make_const (no grad, direct NDArray)
            k = Tensor.make_const(self._k_cache[:, :, :end_pos, :], requires_grad=False)
            v = Tensor.make_const(self._v_cache[:, :, :end_pos, :], requires_grad=False)
        elif self._cache_np_k is not None and self._cache_np_v is not None:
            # CPU mode: use numpy buffers
            k_data = k.numpy()
            v_data = v.numpy()
            self._cache_np_k[:, :, start_pos:end_pos, :] = k_data
            self._cache_np_v[:, :, start_pos:end_pos, :] = v_data
            self._cache_len = end_pos
            k = Tensor(self._cache_np_k[:, :, :end_pos, :], device=self.device, dtype=self.dtype, requires_grad=False)
            v = Tensor(self._cache_np_v[:, :, :end_pos, :], device=self.device, dtype=self.dtype, requires_grad=False)

        total_len = k.shape[2]

        # GQA: repeat k, v
        k = self._repeat_kv(k, self.num_kv_groups)
        v = self._repeat_kv(v, self.num_kv_groups)

        # Attention: Q @ K^T / sqrt(d) — scale as float scalar
        scale = float(self.head_dim ** 0.5)
        attn_weights = _batched_matmul(q, k) / scale

        # Causal mask — built using UniTi ops
        if seq_length > 1:
            # Build upper-triangular mask: positions where token should NOT attend
            # triu_k = total_len - seq_length + 1 means: for prefill, mask future tokens
            triu_k = total_len - seq_length + 1
            # Create mask array: 1.0 where masked, 0.0 where allowed
            mask_data = np.triu(
                np.ones((seq_length, total_len), dtype=np.float32), k=triu_k
            )
            # Convert to large negative where masked
            mask_data = -3.4028235e+38 * mask_data
            mask_t = Tensor(mask_data, device=self.device, dtype=self.dtype, requires_grad=False)
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
    """Embedding layer using numpy index lookup.

    For CUDA devices, caches a CPU-side numpy copy of the embedding table to
    avoid transferring the full weight matrix (e.g. 933 MB for vocab=151936,
    dim=1536) from GPU to CPU on every forward call.  The cache is lazily
    built on first forward and invalidated when weights change (e.g. after
    load_state_dict).
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
        self._weight_numpy_cache = None  # lazy CPU-side cache

    def _invalidate_cache(self):
        """Call after weight is updated (e.g. load_state_dict)."""
        self._weight_numpy_cache = None

    def _get_weight_numpy(self):
        """Return numpy view/copy of embedding weights, using cache for non-numpy devices."""
        if self._weight_numpy_cache is None:
            self._weight_numpy_cache = self.weight.numpy()
        return self._weight_numpy_cache

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (batch_size, seq_len) - float tensor of token IDs
        Returns: (batch_size, seq_len, embedding_dim)

        Note: Embedding lookup requires integer indexing (gather), which is not
        expressible as a differentiable tensor operation. We extract IDs via
        Tensor.numpy() and perform the lookup on weight's underlying data.
        This is equivalent to one_hot(ids) @ weight but without the O(V) memory.
        """
        input_shape = x.shape
        ids = x.numpy().astype(np.int64)
        weight_data = self._get_weight_numpy()
        embedded = weight_data[ids.flatten()].reshape((*input_shape, self.embedding_dim))
        return Tensor(embedded, device=self.device, dtype=self.dtype, requires_grad=False)


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

    def init_cache(self, batch_size: int, max_cache_len: int):
        for layer in self.layers:
            layer.self_attn.init_cache(batch_size, max_cache_len)

    def reset_cache(self):
        for layer in self.layers:
            layer.self_attn.reset_cache()

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
    """Qwen2 model with a language modeling head."""

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
        self.model.init_cache(batch_size, max_cache_len)

    def reset_cache(self):
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
