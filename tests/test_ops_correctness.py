"""
算子正确性验证 — UniTI 前向计算 vs NumPy 参考实现

验证所有核心算子在 cpu_numpy 后端上的前向计算正确性：
  - 每个算子构造随机输入
  - 分别用 UniTI Tensor ops 和 numpy 计算
  - 对比结果, 误差 < 1e-5 视为通过

Usage:
    python tests/test_ops_correctness.py
"""
import sys, os, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

import uniti as uti
from uniti.autograd import Tensor
from uniti import ops

np.random.seed(42)


def check_op(name, uniti_fn, numpy_fn, input_shapes, tol=1e-5, positive=False):
    """Compare UniTI op output with numpy reference."""
    if positive:
        np_inputs = [np.abs(np.random.randn(*s).astype(np.float32)) + 0.1 for s in input_shapes]
    else:
        np_inputs = [np.random.randn(*s).astype(np.float32) for s in input_shapes]

    tensors = [Tensor(x, requires_grad=False) for x in np_inputs]
    uniti_out = uniti_fn(*tensors).realize_cached_data().numpy()
    numpy_out = numpy_fn(*np_inputs)

    max_err = np.max(np.abs(uniti_out - numpy_out))
    rel_err = max_err / (np.max(np.abs(numpy_out)) + 1e-8)
    passed = max_err < tol

    return name, max_err, rel_err, passed


def run_tests():
    print("=" * 70)
    print("  UniTI 算子正确性验证 — 前向计算 vs NumPy 参考实现")
    print("=" * 70)

    results = []
    passed_count = 0
    failed_count = 0

    test_list = [
        # (name, uniti_fn, numpy_fn, shapes, tol, positive)
        ("EWiseAdd",
         lambda a, b: a + b,
         lambda a, b: a + b,
         [(4, 5), (4, 5)], 1e-5, False),

        ("AddScalar",
         lambda a: a + 3.14,
         lambda a: a + 3.14,
         [(4, 5)], 1e-5, False),

        ("EWiseMul",
         lambda a, b: a * b,
         lambda a, b: a * b,
         [(4, 5), (4, 5)], 1e-5, False),

        ("MulScalar",
         lambda a: a * 2.5,
         lambda a: a * 2.5,
         [(4, 5)], 1e-5, False),

        ("EWiseDiv",
         lambda a, b: a / b,
         lambda a, b: a / b,
         [(4, 5), (4, 5)], 1e-4, True),

        ("DivScalar",
         lambda a: a / 3.0,
         lambda a: a / 3.0,
         [(4, 5)], 1e-5, False),

        ("PowerScalar (x^2)",
         lambda a: a ** 2,
         lambda a: a ** 2,
         [(4, 5)], 1e-4, False),

        ("Negate",
         lambda a: -a,
         lambda a: -a,
         [(4, 5)], 1e-5, False),

        ("MatMul",
         lambda a, b: a @ b,
         lambda a, b: a @ b,
         [(3, 4), (4, 5)], 1e-4, False),

        ("Summation(all)",
         lambda a: a.sum(),
         lambda a: np.array([a.sum()]),
         [(4, 5)], 1e-4, False),

        ("Summation(axis=1)",
         lambda a: a.sum(axes=1),
         lambda a: a.sum(axis=1),
         [(4, 5)], 1e-4, False),

        ("Reshape",
         lambda a: a.reshape((10, 2)),
         lambda a: a.reshape((10, 2)),
         [(4, 5)], 1e-5, False),

        ("BroadcastTo",
         lambda a: a.broadcast_to((3, 5)),
         lambda a: np.broadcast_to(a, (3, 5)),
         [(1, 5)], 1e-5, False),

        ("Transpose",
         lambda a: a.transpose(),
         lambda a: a.T,
         [(4, 5)], 1e-5, False),

        ("ReLU",
         lambda a: ops.relu(a),
         lambda a: np.maximum(a, 0),
         [(4, 5)], 1e-5, False),

        ("Tanh",
         lambda a: ops.tanh(a),
         lambda a: np.tanh(a),
         [(4, 5)], 1e-5, False),

        ("Log",
         lambda a: ops.log(a),
         lambda a: np.log(a),
         [(4, 5)], 1e-5, True),

        ("Exp",
         lambda a: ops.exp(a),
         lambda a: np.exp(a),
         [(4, 5)], 1e-4, False),

        ("Sin",
         lambda a: ops.sin(a),
         lambda a: np.sin(a),
         [(4, 5)], 1e-5, False),

        ("Cos",
         lambda a: ops.cos(a),
         lambda a: np.cos(a),
         [(4, 5)], 1e-5, False),

        ("Sqrt",
         lambda a: ops.sqrt(a),
         lambda a: np.sqrt(a),
         [(4, 5)], 1e-5, True),

        ("SiLU",
         lambda a: ops.silu(a),
         lambda a: a / (1 + np.exp(-a)),
         [(4, 5)], 1e-5, False),

        ("LogSumExp(axis=1)",
         lambda a: ops.logsumexp(a, axes=(1,)),
         lambda a: np.log(np.sum(np.exp(a - np.max(a, axis=1, keepdims=True)), axis=1)) + np.max(a, axis=1),
         [(4, 5)], 1e-4, False),

        ("Concatenate",
         lambda a, b: ops.concatenate([a, b], axis=0),
         lambda a, b: np.concatenate([a, b], axis=0),
         [(3, 5), (2, 5)], 1e-5, False),
    ]

    for entry in test_list:
        name, uniti_fn, numpy_fn, shapes = entry[0], entry[1], entry[2], entry[3]
        tol = entry[4] if len(entry) > 4 else 1e-5
        positive = entry[5] if len(entry) > 5 else False
        n, me, re, p = check_op(name, uniti_fn, numpy_fn, shapes, tol, positive)
        if p:
            passed_count += 1
        else:
            failed_count += 1
        results.append((n, me, re, p))
        status = "PASS" if p else "FAIL"
        print(f"  [{'✓' if p else '✗'}] {n:28s}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── nn modules forward ──
    print("\n  --- 神经网络模块前向验证 ---")

    # Linear
    np.random.seed(0)
    w_np = np.random.randn(4, 3).astype(np.float32) * 0.1
    b_np = np.random.randn(1, 3).astype(np.float32) * 0.1
    x_np = np.random.randn(2, 4).astype(np.float32)
    expected = x_np @ w_np + b_np
    linear = uti.nn.Linear(4, 3)
    linear.weight = uti.nn.Parameter(Tensor(w_np, requires_grad=True))
    linear.bias = uti.nn.Parameter(Tensor(b_np, requires_grad=True))
    x_t = Tensor(x_np, requires_grad=False)
    out = linear(x_t).realize_cached_data().numpy()
    me = np.max(np.abs(out - expected))
    p = me < 1e-4
    if p: passed_count += 1
    else: failed_count += 1
    results.append(("nn.Linear", me, me / (np.max(np.abs(expected)) + 1e-8), p))
    print(f"  [{'✓' if p else '✗'}] {'nn.Linear':28s}  max_err={me:.2e}  {'PASS' if p else 'FAIL'}")

    # ReLU Module
    x_np = np.random.randn(3, 4).astype(np.float32)
    expected = np.maximum(x_np, 0)
    out = uti.nn.ReLU()(Tensor(x_np, requires_grad=False)).realize_cached_data().numpy()
    me = np.max(np.abs(out - expected))
    p = me < 1e-5
    if p: passed_count += 1
    else: failed_count += 1
    results.append(("nn.ReLU", me, 0, p))
    print(f"  [{'✓' if p else '✗'}] {'nn.ReLU':28s}  max_err={me:.2e}  {'PASS' if p else 'FAIL'}")

    # LayerNorm1d
    np.random.seed(1)
    x_np = np.random.randn(2, 4).astype(np.float32)
    mean = x_np.mean(axis=1, keepdims=True)
    var = x_np.var(axis=1, keepdims=True)
    expected_norm = (x_np - mean) / np.sqrt(var + 1e-5)
    ln = uti.nn.LayerNorm1d(4)
    # Set weight=1 bias=0 for easier comparison
    ln.w = uti.nn.Parameter(Tensor(np.ones(4, dtype=np.float32), requires_grad=True))
    ln.b = uti.nn.Parameter(Tensor(np.zeros(4, dtype=np.float32), requires_grad=True))
    out = ln(Tensor(x_np, requires_grad=False)).realize_cached_data().numpy()
    me = np.max(np.abs(out - expected_norm))
    p = me < 1e-4
    if p: passed_count += 1
    else: failed_count += 1
    results.append(("nn.LayerNorm1d", me, 0, p))
    print(f"  [{'✓' if p else '✗'}] {'nn.LayerNorm1d':28s}  max_err={me:.2e}  {'PASS' if p else 'FAIL'}")

    # Softmax Loss
    np.random.seed(2)
    logits_np = np.random.randn(4, 10).astype(np.float32)
    y_np = np.array([3, 7, 1, 5])
    y_one_hot = np.zeros((4, 10), dtype=np.float32)
    y_one_hot[np.arange(4), y_np] = 1
    # numpy reference
    log_sum_exp = np.log(np.sum(np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True)), axis=1)) + np.max(logits_np, axis=1)
    correct_logits = logits_np[np.arange(4), y_np]
    expected_loss = np.mean(log_sum_exp - correct_logits)
    loss_fn = uti.nn.SoftmaxLoss()
    loss = loss_fn(Tensor(logits_np, requires_grad=False), Tensor(y_np.astype(np.float32), requires_grad=False))
    loss_val = loss.realize_cached_data().numpy().item()
    me = abs(loss_val - expected_loss)
    p = me < 1e-3
    if p: passed_count += 1
    else: failed_count += 1
    results.append(("nn.SoftmaxLoss", me, 0, p))
    print(f"  [{'✓' if p else '✗'}] {'nn.SoftmaxLoss':28s}  max_err={me:.2e}  {'PASS' if p else 'FAIL'}")

    # ── Summary ──
    print("\n" + "=" * 70)
    total = passed_count + failed_count
    print(f"  总计: {total} 项测试, {passed_count} 通过, {failed_count} 失败")
    print(f"  通过率: {passed_count/total*100:.1f}%")
    print("=" * 70)

    return results, passed_count, failed_count


if __name__ == "__main__":
    t0 = time.time()
    results, passed, failed = run_tests()
    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.2f}s")
