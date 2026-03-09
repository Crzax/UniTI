"""Profile the Python-layer overhead that dominates CUDA decode latency.
The CUDA kernels themselves are fast, but every op goes through:
  1. NDArray.make() -> CudaArray(size) -> cudaMalloc
  2. .compact() -> maybe another cudaMalloc + kernel
  3. pybind11 call overhead
  4. cudaDeviceSynchronize (implicit or explicit)
"""
import sys, time, numpy as np
sys.path.insert(0, 'python')
from uniti.backend_ndarray.ndarray import NDArray, cpu_numpy, cuda
import ctypes

dev_np = cpu_numpy()
dev_cu = cuda()

try:
    libcudart = ctypes.CDLL('libcudart.so')
    def sync(): libcudart.cudaDeviceSynchronize()
except:
    def sync(): pass

HIDDEN = 1536
ITERS = 500

# ---- Test 1: cudaMalloc overhead (NDArray.make) ----
# Every single op (add, mul, etc) calls NDArray.make which calls CudaArray(size) -> cudaMalloc!
print("=== Test 1: Memory allocation overhead ===\n")

sync()
t0 = time.perf_counter()
for _ in range(ITERS):
    arr = NDArray.make((1, HIDDEN), device=dev_cu)
    sync()
t_alloc_cu = (time.perf_counter() - t0) / ITERS

t0 = time.perf_counter()
for _ in range(ITERS):
    arr = NDArray.make((1, HIDDEN), device=dev_np)
t_alloc_np = (time.perf_counter() - t0) / ITERS

print(f"NDArray.make((1,{HIDDEN})):")
print(f"  cpu_numpy: {t_alloc_np*1000:.4f}ms")
print(f"  cuda:      {t_alloc_cu*1000:.4f}ms  ({t_alloc_cu/t_alloc_np:.1f}x slower)")
print()

# ---- Test 2: How many cudaMalloc happen per ewise_add? ----
# ewise_or_scalar does: NDArray.make(output) + maybe 2x compact() 
# Each compact() that triggers does: NDArray.make + compact kernel
print("=== Test 2: Full op overhead (make + call + sync) ===\n")
a_cu = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_cu)
b_cu = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_cu)
a_np = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_np)
b_np = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_np)

# Time WITHOUT sync
t0 = time.perf_counter()
for _ in range(ITERS):
    _ = a_cu + b_cu
t_nosync = (time.perf_counter() - t0) / ITERS

# Time WITH sync after each
sync()
t0 = time.perf_counter()
for _ in range(ITERS):
    _ = a_cu + b_cu
    sync()
t_sync = (time.perf_counter() - t0) / ITERS

t0 = time.perf_counter()
for _ in range(ITERS):
    _ = a_np + b_np
t_np_add = (time.perf_counter() - t0) / ITERS

print(f"ewise_add (1x{HIDDEN}):")
print(f"  cpu_numpy:         {t_np_add*1000:.4f}ms")
print(f"  cuda (no sync):    {t_nosync*1000:.4f}ms  ({t_nosync/t_np_add:.1f}x)")
print(f"  cuda (with sync):  {t_sync*1000:.4f}ms  ({t_sync/t_np_add:.1f}x)")
print(f"  -> kernel launch + cudaMalloc overhead = {t_nosync*1000:.4f}ms")
print(f"  -> sync overhead = {(t_sync-t_nosync)*1000:.4f}ms")
print()

# ---- Test 3: Count operations in one decode step ----
# Every NDArray op = 1 cudaMalloc + 1 kernel launch
# Let's count: per transformer layer:
# RMSNorm: compact_in, make_out, pow(compact+make), sum(compact+make+reduce), add_eps(make),
#          sqrt/pow(make), broadcast(no alloc), div(compact2+make), mul(compact2+make)
#          → ~7-9 ops × 2 = ~16 cudaMalloc+launch per layer for RMSNorm alone
# Linear: matmul(compact2+make) → 1 op but may compact
# Total estimate: ~50-80 CUDA ops per layer → 1400-2240 ops for 28 layers

# Let's measure how much total kernel launch overhead that is
n_ops_estimate = 60 * 28  # ~60 ops per layer × 28 layers
overhead_per_op = t_nosync  # cudaMalloc + kernel launch, no sync

print(f"=== Test 3: Estimated launch overhead per decode step ===\n")
print(f"  Estimated ops per decode step: ~{n_ops_estimate}")
print(f"  Avg overhead per op (no sync): {t_nosync*1000:.4f}ms")
print(f"  Total launch overhead: {n_ops_estimate * overhead_per_op * 1000:.1f}ms")
print(f"  (This alone limits throughput to {1/(n_ops_estimate * overhead_per_op):.1f} tok/s)")
print()

# ---- Test 4: What's inside the overhead? Is it cudaMalloc? ----
# Just time cudaMalloc via creating CudaArrays directly
print("=== Test 4: Pure cudaMalloc cost ===\n")
Array = dev_cu.mod.Array  # CudaArray constructor
sync()
t0 = time.perf_counter()
for _ in range(ITERS):
    a = Array(HIDDEN)  # cudaMalloc(HIDDEN * 4)
    sync()
t_malloc_cu = (time.perf_counter() - t0) / ITERS

# Without sync
t0 = time.perf_counter()
for _ in range(ITERS):
    a = Array(HIDDEN)  # cudaMalloc
t_malloc_nosync = (time.perf_counter() - t0) / ITERS

print(f"CudaArray({HIDDEN}):  {t_malloc_nosync*1000:.4f}ms (no sync), {t_malloc_cu*1000:.4f}ms (with sync)")

# numpy Array
NpArray = dev_np.mod.Array
t0 = time.perf_counter()
for _ in range(ITERS):
    a = NpArray(HIDDEN)
t_malloc_np = (time.perf_counter() - t0) / ITERS
print(f"NumpyArray({HIDDEN}): {t_malloc_np*1000:.4f}ms")
print(f"  -> cudaMalloc is {t_malloc_nosync/t_malloc_np:.1f}x slower than numpy array alloc")
print()

# ---- Test 5: Autograd overhead? ----
# The ops go through Tensor -> ops -> NDArray. 
# How much overhead does the Tensor/autograd layer add?
print("=== Test 5: Tensor (autograd) layer overhead ===\n")
from uniti.autograd import Tensor

ta_cu = Tensor(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_cu, requires_grad=False)
tb_cu = Tensor(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_cu, requires_grad=False)
ta_np = Tensor(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_np, requires_grad=False)
tb_np = Tensor(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_np, requires_grad=False)

sync()
t0 = time.perf_counter()
for _ in range(ITERS):
    _ = ta_cu + tb_cu
    sync()
t_tensor_cu = (time.perf_counter() - t0) / ITERS

t0 = time.perf_counter()
for _ in range(ITERS):
    _ = ta_np + tb_np
t_tensor_np = (time.perf_counter() - t0) / ITERS

print(f"Tensor add (1x{HIDDEN}):")
print(f"  cpu_numpy: {t_tensor_np*1000:.4f}ms")
print(f"  cuda:      {t_tensor_cu*1000:.4f}ms  ({t_tensor_cu/t_tensor_np:.1f}x)")
print(f"  NDArray-only cuda: {t_sync*1000:.4f}ms")
print(f"  -> Tensor layer adds: {(t_tensor_cu - t_sync)*1000:.4f}ms per op")
print()

# ---- Summary ----
print("=" * 70)
print("SUMMARY: Where does the CUDA decode step time go?\n")
# Total per step: ~1000ms (1.0 tok/s actual)
# kernel compute: ~192ms (from bench_cuda_decode.py estimate)
# launch + malloc overhead: ~60*28 * 0.04ms = ~67ms
# actual measurement gap: 1000ms - 192ms = ~808ms unaccounted
# This must be from:
# 1) The Tensor/Python layer processing
# 2) numpy() calls for embedding, KV cache, etc.
# 3) Many more ops than estimated (compact() calls double the count)
tensor_overhead = (t_tensor_cu - t_sync) * n_ops_estimate
malloc_overhead = t_malloc_nosync * n_ops_estimate
kernel_time = 192  # ms from previous benchmark
python_overhead = tensor_overhead + malloc_overhead
print(f"  Kernel compute time (est.):  ~{kernel_time:.0f}ms")
print(f"  cudaMalloc overhead (est.):   ~{malloc_overhead*1000:.0f}ms ({n_ops_estimate} allocs)")
print(f"  Tensor layer overhead (est.): ~{tensor_overhead*1000:.0f}ms")
print(f"  Actual total:                 ~1000ms")
print(f"  Gap (unaccounted):            ~{1000 - kernel_time - malloc_overhead*1000 - tensor_overhead*1000:.0f}ms")
print(f"\nThe gap likely comes from: compact() doubling op count,")
print(f"NDArray(numpy) for cos/sin/mask each step, autograd dispatch, etc.")
