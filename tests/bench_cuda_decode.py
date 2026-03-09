"""Simulate the operations in ONE decode step for DeepSeek-R1-Distill-Qwen-1.5B.
Model config: hidden=1536, heads=12, kv_heads=2, intermediate=8960, layers=28
"""
import sys, time, numpy as np
sys.path.insert(0, 'python')
from uniti.backend_ndarray.ndarray import NDArray, cpu_numpy, cuda
import ctypes

dev_np = cpu_numpy()
dev_cu = cuda()

# Model constants
HIDDEN = 1536
HEADS = 12
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS  # 128
INTER = 8960
LAYERS = 28
KV_GROUPS = HEADS // KV_HEADS  # 6

try:
    libcudart = ctypes.CDLL('libcudart.so')
    def sync(): libcudart.cudaDeviceSynchronize()
except:
    def sync(): pass

def time_op(fn, iters=100):
    fn()  # warmup
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        sync()
    return (time.perf_counter() - t0) / iters

# Per-layer operations during decode (seq_len=1, batch=1)
# For each layer, the key operations are:
# 1. RMSNorm x2: involves sum, power, div, mul on (1, hidden) 
# 2. Linear projections: q_proj(1x1536 @ 1536x1536), k_proj(1x1536 @ 1536x256), 
#    v_proj(1x1536 @ 1536x256), o_proj(1x1536 @ 1536x1536)
# 3. MLP: gate_proj(1x1536 @ 1536x8960), up_proj(1x1536 @ 1536x8960), 
#    down_proj(1x8960 @ 8960x1536)
# 4. Attention: QK^T, softmax, attn@V — all small tensors
# 5. RoPE: ewise_mul + concat + add on (1, heads, 1, head_dim)
# 6. KV cache: setitem + slice + compact

print("=== Timing individual operations for ONE decode step ===\n")

# --- Matmuls (the big ones) ---
# Assume batch=1, seq=1 for decode
x_1h = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_cu)
x_1h_np = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_np)

# Q projection: 1x1536 @ 1536x1536
w_hh = NDArray(np.random.randn(HIDDEN, HIDDEN).astype(np.float32), device=dev_cu)
w_hh_np = NDArray(np.random.randn(HIDDEN, HIDDEN).astype(np.float32), device=dev_np)
t_q_cu = time_op(lambda: x_1h @ w_hh)
t_q_np = time_op(lambda: x_1h_np @ w_hh_np)

# K/V projection: 1x1536 @ 1536x256
w_kv = NDArray(np.random.randn(HIDDEN, KV_HEADS*HEAD_DIM).astype(np.float32), device=dev_cu)
w_kv_np = NDArray(np.random.randn(HIDDEN, KV_HEADS*HEAD_DIM).astype(np.float32), device=dev_np)
t_kv_cu = time_op(lambda: x_1h @ w_kv)
t_kv_np = time_op(lambda: x_1h_np @ w_kv_np)

# Gate/Up projection: 1x1536 @ 1536x8960
w_gate = NDArray(np.random.randn(HIDDEN, INTER).astype(np.float32), device=dev_cu)
w_gate_np = NDArray(np.random.randn(HIDDEN, INTER).astype(np.float32), device=dev_np)
t_gate_cu = time_op(lambda: x_1h @ w_gate)
t_gate_np = time_op(lambda: x_1h_np @ w_gate_np)

# Down projection: 1x8960 @ 8960x1536
x_1i = NDArray(np.random.randn(1, INTER).astype(np.float32), device=dev_cu)
w_down = NDArray(np.random.randn(INTER, HIDDEN).astype(np.float32), device=dev_cu)
x_1i_np = NDArray(np.random.randn(1, INTER).astype(np.float32), device=dev_np)
w_down_np = NDArray(np.random.randn(INTER, HIDDEN).astype(np.float32), device=dev_np)
t_down_cu = time_op(lambda: x_1i @ w_down)
t_down_np = time_op(lambda: x_1i_np @ w_down_np)

print(f"{'Operation':45s} {'numpy':>8s} {'cuda':>8s}  {'ratio':>6s}")
print("-" * 75)
print(f"{'q_proj (1x1536 @ 1536x1536)':45s} {t_q_np*1000:7.3f}ms {t_q_cu*1000:7.3f}ms  {t_q_cu/t_q_np:.2f}x")
print(f"{'k/v_proj (1x1536 @ 1536x256)':45s} {t_kv_np*1000:7.3f}ms {t_kv_cu*1000:7.3f}ms  {t_kv_cu/t_kv_np:.2f}x")
print(f"{'gate/up_proj (1x1536 @ 1536x8960)':45s} {t_gate_np*1000:7.3f}ms {t_gate_cu*1000:7.3f}ms  {t_gate_cu/t_gate_np:.2f}x")
print(f"{'down_proj (1x8960 @ 8960x1536)':45s} {t_down_np*1000:7.3f}ms {t_down_cu*1000:7.3f}ms  {t_down_cu/t_down_np:.2f}x")

# --- Small elementwise ops (decode is ALL small tensors) ---
s_cu = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_cu)
s_cu2 = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_cu)
s_np = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_np)
s_np2 = NDArray(np.random.randn(1, HIDDEN).astype(np.float32), device=dev_np)

t_add_cu = time_op(lambda: s_cu + s_cu2, 500)
t_add_np = time_op(lambda: s_np + s_np2, 500)
t_mul_cu = time_op(lambda: s_cu * s_cu2, 500)
t_mul_np = time_op(lambda: s_np * s_np2, 500)
t_div_cu = time_op(lambda: s_cu / s_cu2, 500)
t_div_np = time_op(lambda: s_np / s_np2, 500)
t_pow_cu = time_op(lambda: s_cu ** 0.5, 500)
t_pow_np = time_op(lambda: s_np ** 0.5, 500)
t_exp_cu = time_op(lambda: s_cu.exp(), 500)
t_exp_np = time_op(lambda: s_np.exp(), 500)
t_smul_cu = time_op(lambda: s_cu * 0.5, 500)
t_smul_np = time_op(lambda: s_np * 0.5, 500)
t_sum_cu = time_op(lambda: s_cu.sum(axis=1), 500)
t_sum_np = time_op(lambda: s_np.sum(axis=1), 500)
t_max_cu = time_op(lambda: s_cu.max(axis=1), 500)
t_max_np = time_op(lambda: s_np.max(axis=1), 500)

print()
print(f"{'ewise_add (1x1536)':45s} {t_add_np*1000:7.3f}ms {t_add_cu*1000:7.3f}ms  {t_add_cu/t_add_np:.2f}x")
print(f"{'ewise_mul (1x1536)':45s} {t_mul_np*1000:7.3f}ms {t_mul_cu*1000:7.3f}ms  {t_mul_cu/t_mul_np:.2f}x")
print(f"{'ewise_div (1x1536)':45s} {t_div_np*1000:7.3f}ms {t_div_cu*1000:7.3f}ms  {t_div_cu/t_div_np:.2f}x")
print(f"{'scalar_mul (1x1536)':45s} {t_smul_np*1000:7.3f}ms {t_smul_cu*1000:7.3f}ms  {t_smul_cu/t_smul_np:.2f}x")
print(f"{'scalar_power^0.5 (1x1536)':45s} {t_pow_np*1000:7.3f}ms {t_pow_cu*1000:7.3f}ms  {t_pow_cu/t_pow_np:.2f}x")
print(f"{'ewise_exp (1x1536)':45s} {t_exp_np*1000:7.3f}ms {t_exp_cu*1000:7.3f}ms  {t_exp_cu/t_exp_np:.2f}x")
print(f"{'reduce_sum (1x1536)':45s} {t_sum_np*1000:7.3f}ms {t_sum_cu*1000:7.3f}ms  {t_sum_cu/t_sum_np:.2f}x")
print(f"{'reduce_max (1x1536)':45s} {t_max_np*1000:7.3f}ms {t_max_cu*1000:7.3f}ms  {t_max_cu/t_max_np:.2f}x")

# --- NDArray creation (happens for cos/sin/mask every step) ---
cos_raw = np.random.randn(1, HEAD_DIM).astype(np.float32)
t_create_cu = time_op(lambda: NDArray(cos_raw, device=dev_cu), 500)
t_create_np = time_op(lambda: NDArray(cos_raw, device=dev_np), 500)
print(f"\n{'NDArray creation (1x128, cos/sin)':45s} {t_create_np*1000:7.3f}ms {t_create_cu*1000:7.3f}ms  {t_create_cu/t_create_np:.2f}x")

# --- Estimate total per decode step ---
print("\n" + "=" * 75)
print("=== Estimated total time per decode step (28 layers) ===\n")

# Per layer matmul count:
# q_proj: 1x (1,1536)@(1536,1536)
# k_proj: 1x (1,1536)@(1536,256) 
# v_proj: 1x (1,1536)@(1536,256)
# o_proj: 1x (1,1536)@(1536,1536)
# gate_proj: 1x (1,1536)@(1536,8960)
# up_proj: 1x (1,1536)@(1536,8960)
# down_proj: 1x (1,8960)@(8960,1536)

matmul_time_cu = (2 * t_q_cu + 2 * t_kv_cu + 2 * t_gate_cu + t_down_cu) * LAYERS
matmul_time_np = (2 * t_q_np + 2 * t_kv_np + 2 * t_gate_np + t_down_np) * LAYERS

# Per layer small ops (rough count from model):
# RMSNorm x2: each has ~5 small ops (mul, sum, pow, div, mul) → 10 ops
# Residual adds: 2
# RoPE: ~6 small ops (slice, neg, concat, mul x2, add)
# Softmax: ~5 ops (max, sub, exp, sum, div)
# Attention overhead: ~4 (scale, GQA repeat, etc)
# MLP: ~3 (silu = sigmoid + mul, final mul)
# Total rough: ~30 small ops per layer
small_ops_per_layer = 30
small_op_avg_cu = (t_add_cu + t_mul_cu + t_div_cu + t_exp_cu + t_smul_cu + t_sum_cu + t_pow_cu + t_max_cu) / 8
small_op_avg_np = (t_add_np + t_mul_np + t_div_np + t_exp_np + t_smul_np + t_sum_np + t_pow_np + t_max_np) / 8
small_ops_time_cu = small_ops_per_layer * LAYERS * small_op_avg_cu
small_ops_time_np = small_ops_per_layer * LAYERS * small_op_avg_np

# NDArray creation per step (cos, sin tensors, mask etc.)
create_per_step = 3  # cos, sin, maybe mask
create_time_cu = LAYERS * create_per_step * t_create_cu
create_time_np = LAYERS * create_per_step * t_create_np

total_cu = matmul_time_cu + small_ops_time_cu + create_time_cu
total_np = matmul_time_np + small_ops_time_np + create_time_np

print(f"{'Component':30s} {'numpy':>10s} {'cuda':>10s}")
print("-" * 55)
print(f"{'Matmul (all projections)':30s} {matmul_time_np*1000:9.1f}ms {matmul_time_cu*1000:9.1f}ms")
print(f"{'Small ops (~30/layer x28)':30s} {small_ops_time_np*1000:9.1f}ms {small_ops_time_cu*1000:9.1f}ms")
print(f"{'NDArray creation (cos/sin)':30s} {create_time_np*1000:9.1f}ms {create_time_cu*1000:9.1f}ms")
print(f"{'ESTIMATED TOTAL':30s} {total_np*1000:9.1f}ms {total_cu*1000:9.1f}ms")
print(f"{'Estimated tok/s':30s} {1/total_np:9.2f}     {1/total_cu:9.2f}")
print(f"\ncuda/numpy ratio: {total_cu/total_np:.2f}x")
