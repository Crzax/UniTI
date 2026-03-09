"""Test OpenBLAS-optimized CPU backend correctness and performance."""
import numpy as np
import time
import sys
sys.path.insert(0, 'python')
from uniti.backend_ndarray.ndarray import NDArray, cpu, cpu_numpy

# === Correctness Test ===
print('=== Correctness Test ===')
a_np = np.random.randn(64, 128).astype(np.float32)
b_np = np.random.randn(128, 64).astype(np.float32)
expected = a_np @ b_np

# cpu backend (should now use OpenBLAS)
a_cpu = NDArray(a_np, device=cpu())
b_cpu = NDArray(b_np, device=cpu())
c_cpu = (a_cpu @ b_cpu).numpy()
err = np.max(np.abs(c_cpu - expected))
print(f'cpu matmul max error: {err:.2e}  (should be < 1e-4)')

# cpu_numpy backend
a_np2 = NDArray(a_np, device=cpu_numpy())
b_np2 = NDArray(b_np, device=cpu_numpy())
c_np2 = (a_np2 @ b_np2).numpy()
err2 = np.max(np.abs(c_np2 - expected))
print(f'cpu_numpy matmul max error: {err2:.2e}')

# Test non-tile-aligned sizes (uses Matmul, not MatmulTiled)
print('\n--- Non-tile-aligned (e.g. 13x17 @ 17x11) ---')
a_s = np.random.randn(13, 17).astype(np.float32)
b_s = np.random.randn(17, 11).astype(np.float32)
exp_s = a_s @ b_s
c_s = (NDArray(a_s, device=cpu()) @ NDArray(b_s, device=cpu())).numpy()
err_s = np.max(np.abs(c_s - exp_s))
print(f'Non-aligned matmul max error: {err_s:.2e}')

# === Performance Test ===
print('\n=== Performance Test (1536x1536 matmul) ===')
m, n, p = 1536, 1536, 1536
a_big = np.random.randn(m, n).astype(np.float32)
b_big = np.random.randn(n, p).astype(np.float32)

# Warmup
a_c = NDArray(a_big, device=cpu())
b_c = NDArray(b_big, device=cpu())
_ = a_c @ b_c

# cpu backend timing
times = []
for _ in range(5):
    a_c = NDArray(a_big, device=cpu())
    b_c = NDArray(b_big, device=cpu())
    t0 = time.perf_counter()
    c = a_c @ b_c
    t1 = time.perf_counter()
    times.append(t1 - t0)
cpu_time = np.median(times)
print(f'cpu (OpenBLAS):  {cpu_time*1000:.1f} ms')

# cpu_numpy backend timing
times = []
for _ in range(5):
    a_n = NDArray(a_big, device=cpu_numpy())
    b_n = NDArray(b_big, device=cpu_numpy())
    t0 = time.perf_counter()
    c = a_n @ b_n
    t1 = time.perf_counter()
    times.append(t1 - t0)
numpy_time = np.median(times)
print(f'cpu_numpy:       {numpy_time*1000:.1f} ms')

# Pure numpy reference
times = []
for _ in range(5):
    t0 = time.perf_counter()
    c = a_big @ b_big
    t1 = time.perf_counter()
    times.append(t1 - t0)
ref_time = np.median(times)
print(f'pure numpy:      {ref_time*1000:.1f} ms')

print(f'\ncpu / cpu_numpy ratio: {cpu_time/numpy_time:.2f}x')
print(f'cpu / pure numpy ratio: {cpu_time/ref_time:.2f}x')

# Also test a smaller size typical for inference (batch=1)
print('\n=== Performance Test (1x1536 @ 1536x1536, typical inference) ===')
a_inf = np.random.randn(1, 1536).astype(np.float32)
b_inf = np.random.randn(1536, 1536).astype(np.float32)

times_cpu = []
times_np = []
for _ in range(20):
    ac = NDArray(a_inf, device=cpu())
    bc = NDArray(b_inf, device=cpu())
    t0 = time.perf_counter()
    _ = ac @ bc
    t1 = time.perf_counter()
    times_cpu.append(t1 - t0)

    an = NDArray(a_inf, device=cpu_numpy())
    bn = NDArray(b_inf, device=cpu_numpy())
    t0 = time.perf_counter()
    _ = an @ bn
    t1 = time.perf_counter()
    times_np.append(t1 - t0)

print(f'cpu (OpenBLAS):  {np.median(times_cpu)*1000:.2f} ms')
print(f'cpu_numpy:       {np.median(times_np)*1000:.2f} ms')
print('\nAll tests done!')
