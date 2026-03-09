"""Test lm_head matmul cost: 1x1536 @ 1536x151936 (vocab_size)"""
import sys, time, numpy as np
sys.path.insert(0, 'python')
from uniti.backend_ndarray.ndarray import NDArray, cpu_numpy, cuda
import ctypes

try:
    libcudart = ctypes.CDLL('libcudart.so')
    def sync(): libcudart.cudaDeviceSynchronize()
except:
    def sync(): pass

HIDDEN = 1536
VOCAB = 151936

dev_np = cpu_numpy()
dev_cu = cuda()

x_np = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_np)
w_np = NDArray(np.random.randn(HIDDEN, VOCAB).astype(np.float32), device=dev_np)

x_cu = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_cu)
w_cu = NDArray(np.random.randn(HIDDEN, VOCAB).astype(np.float32), device=dev_cu)

# Warmup
_ = x_np @ w_np
sync()
_ = x_cu @ w_cu
sync()

# numpy
t0 = time.perf_counter()
for _ in range(5):
    _ = x_np @ w_np
t_np = (time.perf_counter() - t0) / 5

# cuda 
sync()
t0 = time.perf_counter()
for _ in range(5):
    _ = x_cu @ w_cu
    sync()
t_cu = (time.perf_counter() - t0) / 5

print(f"lm_head matmul (1x{HIDDEN} @ {HIDDEN}x{VOCAB}):")
print(f"  cpu_numpy: {t_np*1000:.1f}ms")
print(f"  cuda:      {t_cu*1000:.1f}ms  ({t_cu/t_np:.2f}x)")
print()

# Also test embedding lookup overhead (involves to_numpy on CUDA)
print(f"Embedding weight to_numpy (download {VOCAB}x{HIDDEN} from GPU):")
big_cu = NDArray(np.random.randn(VOCAB, HIDDEN).astype(np.float32), device=dev_cu)
sync()
t0 = time.perf_counter()
for _ in range(3):
    np_data = big_cu.numpy()
t_dl = (time.perf_counter() - t0) / 3
print(f"  {t_dl*1000:.1f}ms per download ({VOCAB*HIDDEN*4/1e6:.0f}MB)")
print()

# Test logits to_numpy (download 1x151936 from GPU — needed for argmax)
logits_cu = NDArray(np.random.randn(1, 1, VOCAB).astype(np.float32), device=dev_cu)
sync()
t0 = time.perf_counter()
for _ in range(20):
    np_logits = logits_cu.numpy()
t_logits_dl = (time.perf_counter() - t0) / 20
print(f"logits download (1x1x{VOCAB} from GPU): {t_logits_dl*1000:.2f}ms")
print()

# Test embedding: weight.numpy() + index lookup (happens every step for CUDA!)
# On CUDA, QwenEmbedding.forward() does:
#   self.weight.numpy() -> downloads ENTIRE 151936x1536 matrix from GPU!
#   Then does numpy index lookup
#   Then NDArray(result, device=cuda) -> uploads result back to GPU
print(f"=== Embedding forward simulation (CUDA) ===")
print(f"  Step 1: weight.numpy() = download {VOCAB}x{HIDDEN} float32 = {VOCAB*HIDDEN*4/1e6:.0f}MB")
sync()
t0 = time.perf_counter()
_ = big_cu.numpy()
t_step1 = time.perf_counter() - t0
print(f"    Time: {t_step1*1000:.1f}ms")

print(f"  Step 2: numpy index lookup")
wdata = np.random.randn(VOCAB, HIDDEN).astype(np.float32)
t0 = time.perf_counter()
_ = wdata[np.array([9707])]
t_step2 = time.perf_counter() - t0
print(f"    Time: {t_step2*1000:.4f}ms")

print(f"  Step 3: upload result (1x1x{HIDDEN}) to GPU")
small = np.random.randn(1, 1, HIDDEN).astype(np.float32)
sync()
t0 = time.perf_counter()
_ = NDArray(small, device=dev_cu)
sync()
t_step3 = time.perf_counter() - t0
print(f"    Time: {t_step3*1000:.2f}ms")

print(f"\n  TOTAL embedding (CUDA): {(t_step1+t_step2+t_step3)*1000:.1f}ms PER DECODE STEP!")
print(f"  (This downloads the ENTIRE {VOCAB*HIDDEN*4/1e6:.0f}MB embedding table every single token!)")

# Total estimated decode step time for CUDA
layer_time = 8.6  # ms, from profile
lm_head_time = t_cu * 1000
embed_time = (t_step1 + t_step2 + t_step3) * 1000
logits_dl_time = t_logits_dl * 1000

print(f"\n{'='*60}")
print(f"=== CUDA decode step time breakdown ===\n")
print(f"  28 transformer layers:  {layer_time*28:.0f}ms")
print(f"  lm_head matmul:         {lm_head_time:.0f}ms")
print(f"  Embedding (GPU→CPU→GPU): {embed_time:.0f}ms  <-- THE KILLER!")
print(f"  logits download:        {logits_dl_time:.1f}ms")
print(f"  Estimated total:        {layer_time*28 + lm_head_time + embed_time + logits_dl_time:.0f}ms")
print(f"  Actual measured:        ~1000ms")
