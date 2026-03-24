"""Benchmark: cuda vs cpu_numpy backend, per-operation.
Finds where CUDA is slower than numpy (CPU) and why.
"""
import sys, time, numpy as np
sys.path.insert(0, 'python')
from uniti.backend_ndarray.ndarray import NDArray, cpu_numpy

# Try to import cuda
try:
    from uniti.backend_ndarray.ndarray import cuda
    dev_cu = cuda()
    if not dev_cu.enabled():
        print("CUDA device not available!")
        sys.exit(1)
    print(f"CUDA device: {dev_cu}")
except Exception as e:
    print(f"Cannot load CUDA: {e}")
    sys.exit(1)

dev_np = cpu_numpy()
N = 1536

def sync_cuda():
    """Ensure all CUDA operations are finished."""
    import ctypes
    try:
        libcudart = ctypes.CDLL('libcudart.so')
        libcudart.cudaDeviceSynchronize()
    except:
        pass  # fallback: hope it's sync enough

def bench(name, fn_np, fn_cu, iters=50, sync=True):
    # warmup
    fn_np(); fn_cu()
    if sync: sync_cuda()

    t0 = time.perf_counter()
    for _ in range(iters): fn_np()
    t_np = (time.perf_counter() - t0) / iters

    if sync: sync_cuda()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn_cu()
        if sync: sync_cuda()
    t_cu = (time.perf_counter() - t0) / iters

    ratio = t_cu / t_np if t_np > 0 else float('inf')
    faster = "numpy WINS" if ratio > 1.2 else ("cuda WINS" if ratio < 0.8 else "~same")
    print(f'{name:50s}  numpy={t_np*1000:7.2f}ms  cuda={t_cu*1000:7.2f}ms  ratio={ratio:.2f}x  [{faster}]')

# Prepare data
a_np_raw = np.random.randn(1, N).astype(np.float32)
b_np_raw = np.random.randn(N, N).astype(np.float32)
big_raw = np.random.randn(N, N).astype(np.float32)

a_np = NDArray(a_np_raw, device=dev_np); b_np = NDArray(b_np_raw, device=dev_np)
a_cu = NDArray(a_np_raw, device=dev_cu); b_cu = NDArray(b_np_raw, device=dev_cu)
x_np = NDArray(big_raw, device=dev_np); y_np = NDArray(big_raw, device=dev_np)
x_cu = NDArray(big_raw, device=dev_cu); y_cu = NDArray(big_raw, device=dev_cu)

print(f"\n=== Per-operation benchmark: cpu_numpy vs cuda, size={N} ===\n")

# --- Matmul: the most important operation ---
bench(f"matmul (1x{N} @ {N}x{N})", lambda: a_np @ b_np, lambda: a_cu @ b_cu, 20)

# Also test bigger batch matmul
a2_np_raw = np.random.randn(N, N).astype(np.float32)
a2_np = NDArray(a2_np_raw, device=dev_np); a2_cu = NDArray(a2_np_raw, device=dev_cu)
bench(f"matmul ({N}x{N} @ {N}x{N})", lambda: a2_np @ b_np, lambda: a2_cu @ b_cu, 10)

# --- Elementwise ops ---
bench(f"ewise_add ({N}x{N})", lambda: x_np + y_np, lambda: x_cu + y_cu)
bench(f"ewise_mul ({N}x{N})", lambda: x_np * y_np, lambda: x_cu * y_cu)
bench(f"ewise_div ({N}x{N})", lambda: x_np / y_np, lambda: x_cu / y_cu)
bench(f"scalar_mul ({N}x{N})", lambda: x_np * 0.5, lambda: x_cu * 0.5)
# Use non-negative data for power^0.5 to avoid NaN from sqrt of negative numbers
pos_raw = np.abs(big_raw)
p_np = NDArray(pos_raw, device=dev_np); p_cu = NDArray(pos_raw, device=dev_cu)
bench(f"scalar_power^0.5 ({N}x{N})", lambda: p_np ** 0.5, lambda: p_cu ** 0.5)
bench(f"ewise_exp ({N}x{N})", lambda: x_np.exp(), lambda: x_cu.exp())

# --- Reductions ---
bench(f"reduce_sum (axis=1, {N}x{N})", lambda: x_np.sum(axis=1), lambda: x_cu.sum(axis=1))
bench(f"reduce_max (axis=1, {N}x{N})", lambda: x_np.max(axis=1), lambda: x_cu.max(axis=1))

# --- Compact (permute then compact - attention path) ---
big4d_raw = np.random.randn(1, 12, 128, 128).astype(np.float32)
big4d_np = NDArray(big4d_raw, device=dev_np)
big4d_cu = NDArray(big4d_raw, device=dev_cu)
perm_np = big4d_np.permute((0, 2, 1, 3))
perm_cu = big4d_cu.permute((0, 2, 1, 3))
bench("compact (1,128,12,128 permuted)", lambda: perm_np.compact(), lambda: perm_cu.compact(), 20)

# --- NDArray creation (CPU->GPU transfer) ---
raw = np.random.randn(N, N).astype(np.float32)
bench(f"NDArray creation ({N}x{N})", lambda: NDArray(raw, device=dev_np), lambda: NDArray(raw, device=dev_cu), 10)

# --- What really matters: simulate a single decode step's overhead ---
# Each decode step does: lots of small ops on (1, seq_len) shaped tensors
# Test small tensor overhead
small = np.random.randn(1, N).astype(np.float32)
sx_np = NDArray(small, device=dev_np); sy_np = NDArray(small, device=dev_np)
sx_cu = NDArray(small, device=dev_cu); sy_cu = NDArray(small, device=dev_cu)
print(f"\n=== Small tensor (decode-step-like) ops, shape=(1, {N}) ===\n")
bench(f"ewise_add (1x{N})", lambda: sx_np + sy_np, lambda: sx_cu + sy_cu, 200)
bench(f"ewise_mul (1x{N})", lambda: sx_np * sy_np, lambda: sx_cu * sy_cu, 200)
bench(f"scalar_mul (1x{N})", lambda: sx_np * 0.5, lambda: sx_cu * 0.5, 200)
# Use non-negative data for power^0.5
sp_raw = np.abs(small)
sp_np = NDArray(sp_raw, device=dev_np); sp_cu = NDArray(sp_raw, device=dev_cu)
bench(f"scalar_power^0.5 (1x{N})", lambda: sp_np ** 0.5, lambda: sp_cu ** 0.5, 200)
bench(f"ewise_exp (1x{N})", lambda: sx_np.exp(), lambda: sx_cu.exp(), 200)
bench(f"reduce_sum (1x{N})", lambda: sx_np.sum(axis=1), lambda: sx_cu.sum(axis=1), 200)

# KV cache update simulation: setitem on a (1, 12, max_len, 128) shaped array
max_len = 256
kv_shape_raw = np.zeros((1, 12, max_len, 128), dtype=np.float32)
kv_np = NDArray(kv_shape_raw, device=dev_np)
kv_cu = NDArray(kv_shape_raw, device=dev_cu)
new_kv_raw = np.random.randn(1, 12, 1, 128).astype(np.float32)
new_kv_np = NDArray(new_kv_raw, device=dev_np)
new_kv_cu = NDArray(new_kv_raw, device=dev_cu)
print(f"\n=== KV cache setitem (1,12,1,128 into 1,12,{max_len},128) ===\n")
bench("KV cache setitem", 
      lambda: kv_np.__setitem__((slice(None), slice(None), slice(10, 11), slice(None)), new_kv_np),
      lambda: kv_cu.__setitem__((slice(None), slice(None), slice(10, 11), slice(None)), new_kv_cu),
      100)

# KV cache slice simulation
bench("KV cache slice+compact (1,12,:50,128)",
      lambda: kv_np[:, :, :50, :].compact(),
      lambda: kv_cu[:, :, :50, :].compact(),
      50)
