"""Profile: why is cpu_numpy slightly slower than before (1.20 vs 1.23)?"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python'))
from uniti.backend_ndarray.ndarray import NDArray, cpu_numpy
from uniti.backend_ndarray import ndarray_backend_numpy as nb

dev = cpu_numpy()

# Test 1: KV cache slice contiguity
print("=== KV Cache Slice Contiguity ===")
cache = np.zeros((1, 2, 100, 128), dtype=np.float32)
s = cache[:, :, :10, :]
print(f"full cache C_CONTIGUOUS: {cache.flags['C_CONTIGUOUS']}")
print(f"cache[:,:,:10,:] C_CONTIGUOUS: {s.flags['C_CONTIGUOUS']}")
print(f"  -> Zero-copy? {s.flags['C_CONTIGUOUS'] and s.dtype == np.float32}")

# Test 2: Compare NDArray creation speed: from_numpy vs zero-copy
print("\n=== NDArray Creation Speed ===")
a_full = np.random.randn(1, 2, 10, 128).astype(np.float32)

# Zero-copy path (C-contiguous float32)
t0 = time.perf_counter()
for _ in range(10000):
    nd = NDArray(a_full, device=dev)
t1 = time.perf_counter()
print(f"C-contiguous float32 (zero-copy): {(t1-t0)*1000/10:.2f} us / call")

# Non-C-contiguous (from slice) - falls to from_numpy path
a_slice = cache[:, :, :10, :]
t0 = time.perf_counter()
for _ in range(10000):
    nd = NDArray(a_slice, device=dev)
t1 = time.perf_counter()
print(f"Non-C-contiguous slice (from_numpy): {(t1-t0)*1000/10:.2f} us / call")

# from_numpy path (original, pre-optimization)
t0 = time.perf_counter()
for _ in range(10000):
    arr = NDArray.make(a_full.shape, device=dev)
    dev.from_numpy(np.ascontiguousarray(a_full), arr._handle)
t1 = time.perf_counter()
print(f"Old from_numpy path (explicit): {(t1-t0)*1000/10:.2f} us / call")

# Test 3: Is the slice ascontiguousarray() expensive?
print("\n=== ascontiguousarray Cost ===")
big_cache = np.zeros((1, 2, 200, 128), dtype=np.float32)
for n in [1, 5, 10, 50]:
    s = big_cache[:, :, :n, :]
    t0 = time.perf_counter()
    for _ in range(10000):
        c = np.ascontiguousarray(s)
    t1 = time.perf_counter()
    print(f"  ascontiguousarray[:,:,:{n},:]: {(t1-t0)*1000/10:.2f} us / call, contiguous={s.flags['C_CONTIGUOUS']}")
