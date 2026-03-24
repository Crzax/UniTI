"""Benchmark: cpu_numpy vs cpu (C++) backend, per-operation."""
import sys, time, numpy as np
sys.path.insert(0, 'python')
from uniti.backend_ndarray.ndarray import NDArray, cpu, cpu_numpy

N = 1536
dev_np = cpu_numpy()
dev_c = cpu()

def bench(name, fn_np, fn_c, iters=50):
    # warmup
    fn_np(); fn_c()
    t0 = time.perf_counter()
    for _ in range(iters): fn_np()
    t_np = (time.perf_counter() - t0) / iters
    t0 = time.perf_counter()
    for _ in range(iters): fn_c()
    t_c = (time.perf_counter() - t0) / iters
    ratio = t_c / t_np if t_np > 0 else float('inf')
    print(f'{name:45s}  numpy={t_np*1000:7.2f}ms  cpu={t_c*1000:7.2f}ms  ratio={ratio:.2f}x')

# Prepare data
a_np = np.random.randn(1, N).astype(np.float32)
b_np = np.random.randn(N, N).astype(np.float32)
big = np.random.randn(N, N).astype(np.float32)

a1 = NDArray(a_np, device=dev_np); b1 = NDArray(b_np, device=dev_np)
a2 = NDArray(a_np, device=dev_c);  b2 = NDArray(b_np, device=dev_c)
x_np = NDArray(big, device=dev_np); y_np = NDArray(big, device=dev_np)
x_c  = NDArray(big, device=dev_c);  y_c  = NDArray(big, device=dev_c)

print(f"=== Per-operation benchmark: cpu_numpy vs cpu(C++), size={N} ===\n")

bench(f"matmul (1x{N} @ {N}x{N})", lambda: a1 @ b1, lambda: a2 @ b2, 20)
bench(f"ewise_add ({N}x{N})", lambda: x_np + y_np, lambda: x_c + y_c)
bench(f"ewise_mul ({N}x{N})", lambda: x_np * y_np, lambda: x_c * y_c)
bench(f"ewise_div ({N}x{N})", lambda: x_np / y_np, lambda: x_c / y_c)
bench(f"scalar_mul ({N}x{N})", lambda: x_np * 0.5, lambda: x_c * 0.5)
# Use non-negative data for power^0.5 to avoid NaN from sqrt of negative numbers
pos = np.abs(big)
p_np = NDArray(pos, device=dev_np); p_c = NDArray(pos, device=dev_c)
bench(f"scalar_power^0.5 ({N}x{N})", lambda: p_np ** 0.5, lambda: p_c ** 0.5)
bench(f"ewise_exp ({N}x{N})", lambda: x_np.exp(), lambda: x_c.exp())
bench(f"reduce_sum (axis=1, {N}x{N})", lambda: x_np.sum(axis=1), lambda: x_c.sum(axis=1))
bench(f"reduce_max (axis=1, {N}x{N})", lambda: x_np.max(axis=1), lambda: x_c.max(axis=1))

# compact (permute then compact - what attention does)
big4d_np = NDArray(np.random.randn(1, 12, 128, 128).astype(np.float32), device=dev_np)
big4d_c  = NDArray(np.random.randn(1, 12, 128, 128).astype(np.float32), device=dev_c)
perm_np = big4d_np.permute((0, 2, 1, 3))
perm_c  = big4d_c.permute((0, 2, 1, 3))
bench("compact (1,128,12,128 permuted)", lambda: perm_np.compact(), lambda: perm_c.compact(), 20)

# NDArray creation (simulates weight loading / tensor creation)
raw = np.random.randn(N, N).astype(np.float32)
bench(f"NDArray creation ({N}x{N})", lambda: NDArray(raw, device=dev_np), lambda: NDArray(raw, device=dev_c), 10)
