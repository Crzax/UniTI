"""Profile a single decode step on CUDA — 1 layer only for speed."""
import sys, os, time, numpy as np
sys.path.insert(0, 'python')

from uniti.autograd import Tensor, no_grad
from uniti.nn.nn_qwen2 import Qwen2DecoderLayer, RMSNorm, QwenEmbedding
from uniti.backend_selection import cuda, cpu_numpy
from uniti import ops
import ctypes

try:
    libcudart = ctypes.CDLL('libcudart.so')
    def sync(): libcudart.cudaDeviceSynchronize()
except:
    def sync(): pass

HIDDEN = 1536
HEADS = 12
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS
INTER = 8960

device = cuda()
print(f"Device: {device}", flush=True)

# Create a single decoder layer  
print("Creating 1 decoder layer...", flush=True)
layer = Qwen2DecoderLayer(
    hidden_size=HIDDEN, num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS, intermediate_size=INTER,
    rms_norm_eps=1e-6, max_position_embeddings=4096,
    rope_theta=10000.0, device=device, dtype='float32',
)
print("Layer created!", flush=True)

# Init KV cache
layer.self_attn.init_cache(1, 128)

# Prefill one token
with no_grad():
    x_pf = Tensor(np.random.randn(1, 1, HIDDEN).astype(np.float32), device=device, requires_grad=False)
    sync()
    _ = layer(x_pf, start_pos=0)
    sync()
    print("Prefill done.", flush=True)

# Decode step
with no_grad():
    x_dec = Tensor(np.random.randn(1, 1, HIDDEN).astype(np.float32), device=device, requires_grad=False)
    sync()
    t0 = time.perf_counter()
    _ = layer(x_dec, start_pos=1)
    sync()
    t1 = time.perf_counter()
    print(f"1-layer decode: {(t1-t0)*1000:.1f}ms (×28 = {(t1-t0)*28*1000:.0f}ms)", flush=True)

# Profile
with no_grad():
    x_dec = Tensor(np.random.randn(1, 1, HIDDEN).astype(np.float32), device=device, requires_grad=False)
    sync()
    
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    _ = layer(x_dec, start_pos=2)
    sync()
    pr.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print("\n--- Top functions by tottime ---")
    print(s.getvalue())

# Also do same on cpu_numpy for comparison
print("\n=== Now same on cpu_numpy ===\n", flush=True)
layer_np = Qwen2DecoderLayer(
    hidden_size=HIDDEN, num_attention_heads=HEADS,
    num_key_value_heads=KV_HEADS, intermediate_size=INTER,
    rms_norm_eps=1e-6, max_position_embeddings=4096,
    rope_theta=10000.0, device=cpu_numpy(), dtype='float32',
)
layer_np.self_attn.init_cache(1, 128)
with no_grad():
    x_pf_np = Tensor(np.random.randn(1, 1, HIDDEN).astype(np.float32), device=cpu_numpy(), requires_grad=False)
    _ = layer_np(x_pf_np, start_pos=0)
    
    x_dec_np = Tensor(np.random.randn(1, 1, HIDDEN).astype(np.float32), device=cpu_numpy(), requires_grad=False)
    t0 = time.perf_counter()
    _ = layer_np(x_dec_np, start_pos=1)
    t1 = time.perf_counter()
    print(f"1-layer decode (numpy): {(t1-t0)*1000:.1f}ms (×28 = {(t1-t0)*28*1000:.0f}ms)", flush=True)

    x_dec_np2 = Tensor(np.random.randn(1, 1, HIDDEN).astype(np.float32), device=cpu_numpy(), requires_grad=False)
    pr2 = cProfile.Profile()
    pr2.enable()
    _ = layer_np(x_dec_np2, start_pos=2)
    pr2.disable()
    
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr2, stream=s2).sort_stats('tottime')
    ps2.print_stats(30)
    print("\n--- Top functions by tottime (numpy) ---")
    print(s2.getvalue())
