"""Detailed matmul timing to find where overhead is."""
import numpy as np, time, sys
sys.path.insert(0, 'python')
from uniti.backend_ndarray.ndarray import NDArray, cpu, cpu_numpy

# Pre-create arrays 
a = np.random.randn(1, 1536).astype(np.float32)
b = np.random.randn(1536, 1536).astype(np.float32)

dev = cpu()
ac = NDArray(a, device=dev)
bc = NDArray(b, device=dev)
print('a is compact:', ac.is_compact())
print('b is compact:', bc.is_compact())

# 1) Pure sgemm call only
ac_c = ac.compact()
bc_c = bc.compact()
times = []
for _ in range(50):
    out = NDArray.make((1, 1536), device=dev)
    t0 = time.perf_counter()
    dev.matmul(ac_c._handle, bc_c._handle, out._handle, 1, 1536, 1536)
    t1 = time.perf_counter()
    times.append(t1 - t0)
print(f'Pure sgemm (1x1536 @ 1536x1536): {np.median(times)*1000:.2f} ms')

# 2) Full __matmul__ call
times2 = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = ac @ bc
    t1 = time.perf_counter()
    times2.append(t1 - t0)
print(f'Full __matmul__: {np.median(times2)*1000:.2f} ms')

# 3) cpu_numpy comparison 
an = NDArray(a, device=cpu_numpy())
bn = NDArray(b, device=cpu_numpy())
times3 = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = an @ bn
    t1 = time.perf_counter()
    times3.append(t1 - t0)
print(f'cpu_numpy __matmul__: {np.median(times3)*1000:.2f} ms')

# 4) Pure numpy
times4 = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = a @ b
    t1 = time.perf_counter()
    times4.append(t1 - t0)
print(f'Pure numpy: {np.median(times4)*1000:.2f} ms')

# 5) Larger 1536x1536
print('\n--- 1536x1536 @ 1536x1536 ---')
a2 = np.random.randn(1536, 1536).astype(np.float32)
b2 = np.random.randn(1536, 1536).astype(np.float32)
ac2 = NDArray(a2, device=dev).compact()
bc2 = NDArray(b2, device=dev).compact()
times5 = []
for _ in range(5):
    out = NDArray.make((1536, 1536), device=dev)
    t0 = time.perf_counter()
    dev.matmul(ac2._handle, bc2._handle, out._handle, 1536, 1536, 1536)
    t1 = time.perf_counter()
    times5.append(t1 - t0)
print(f'Pure sgemm: {np.median(times5)*1000:.1f} ms')

times6 = []
for _ in range(5):
    t0 = time.perf_counter()
    _ = a2 @ b2
    t1 = time.perf_counter()
    times6.append(t1 - t0)
print(f'Pure numpy: {np.median(times6)*1000:.1f} ms')
