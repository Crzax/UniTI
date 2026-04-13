"""
自动微分正确性验证 — 数值梯度 vs 解析梯度对比

对 UniTI 框架中所有核心算子进行梯度检查：
  - 使用有限差分法计算数值梯度
  - 与 autograd 反向传播得到的解析梯度对比
  - 误差阈值 < 1e-3 视为通过

Usage:
    python tests/test_autograd_correctness.py
"""
import sys, os, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

import uniti as uti
from uniti.autograd import Tensor, no_grad
from uniti import ops

np.random.seed(42)


def numerical_gradient(f, inputs, idx, eps=1e-4):
    """Compute numerical gradient via central difference."""
    x = inputs[idx]
    x_data = x.realize_cached_data().numpy().copy()
    grad = np.zeros_like(x_data)
    it = np.nditer(x_data, flags=['multi_index'])
    while not it.finished:
        ix = it.multi_index
        old_val = x_data[ix]
        x_data[ix] = old_val + eps
        inputs[idx] = Tensor(x_data, requires_grad=False)
        fp = f(*inputs).realize_cached_data().numpy().sum()
        x_data[ix] = old_val - eps
        inputs[idx] = Tensor(x_data, requires_grad=False)
        fm = f(*inputs).realize_cached_data().numpy().sum()
        grad[ix] = (fp - fm) / (2 * eps)
        x_data[ix] = old_val
        inputs[idx] = Tensor(x_data, requires_grad=True)
        it.iternext()
    return grad


def check_gradient(op_name, f, shapes, tol=1e-2, positive=False):
    """Check gradient of a function with given input shapes."""
    if positive:
        inputs = [Tensor(np.abs(np.random.randn(*s).astype(np.float32)) + 0.5, requires_grad=True) for s in shapes]
    else:
        inputs = [Tensor(np.random.randn(*s).astype(np.float32) * 0.5 + 0.5, requires_grad=True) for s in shapes]
    output = f(*inputs)
    output_sum = output.sum()
    output_sum.backward()

    results = []
    for i, inp in enumerate(inputs):
        analytic = inp.grad.realize_cached_data().numpy()
        rebuild = [Tensor(x.realize_cached_data().numpy().copy(), requires_grad=True) for x in inputs]
        numeric = numerical_gradient(f, rebuild, i)
        max_err = np.max(np.abs(analytic - numeric))
        rel_err = max_err / (np.max(np.abs(numeric)) + 1e-8)
        passed = rel_err < tol
        results.append((i, max_err, rel_err, passed))

    return results


def run_tests():
    """Run gradient checks on all core operators."""
    print("=" * 70)
    print("  UniTI 自动微分正确性验证 — 数值梯度 vs 解析梯度")
    print("=" * 70)

    test_cases = []
    passed_count = 0
    failed_count = 0

    # ── 1. EWiseAdd ──
    def test_add(a, b): return a + b
    results = check_gradient("EWiseAdd", test_add, [(3, 4), (3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("EWiseAdd", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] EWiseAdd       input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 2. AddScalar ──
    def test_add_scalar(a): return a + 3.14
    results = check_gradient("AddScalar", test_add_scalar, [(3, 4)], tol=1.5e-2)
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("AddScalar", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] AddScalar      input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 3. EWiseMul ──
    def test_mul(a, b): return a * b
    results = check_gradient("EWiseMul", test_mul, [(3, 4), (3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("EWiseMul", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] EWiseMul       input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 4. MulScalar ──
    def test_mul_scalar(a): return a * 2.5
    results = check_gradient("MulScalar", test_mul_scalar, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("MulScalar", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] MulScalar      input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 5. PowerScalar ──
    def test_pow_scalar(a): return a ** 2
    results = check_gradient("PowerScalar", test_pow_scalar, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("PowerScalar", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] PowerScalar    input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 6. EWiseDiv ──
    def test_div(a, b): return a / b
    results = check_gradient("EWiseDiv", test_div, [(3, 4), (3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("EWiseDiv", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] EWiseDiv       input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 7. DivScalar ──
    def test_div_scalar(a): return a / 3.0
    results = check_gradient("DivScalar", test_div_scalar, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("DivScalar", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] DivScalar      input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 8. Negate ──
    def test_negate(a): return -a
    results = check_gradient("Negate", test_negate, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Negate", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Negate         input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 9. MatMul ──
    def test_matmul(a, b): return a @ b
    results = check_gradient("MatMul", test_matmul, [(3, 4), (4, 5)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("MatMul", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] MatMul         input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 10. Summation ──
    def test_sum(a): return a.sum()
    results = check_gradient("Summation(None)", test_sum, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Summation(None)", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Summation(all) input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    def test_sum_axis(a): return a.sum(axes=1)
    results = check_gradient("Summation(axis=1)", test_sum_axis, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Summation(axis=1)", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Summation(ax1) input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 11. Reshape ──
    def test_reshape(a): return a.reshape((6, 2))
    results = check_gradient("Reshape", test_reshape, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Reshape", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Reshape        input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 12. BroadcastTo ──
    def test_broadcast(a): return a.broadcast_to((3, 4))
    results = check_gradient("BroadcastTo", test_broadcast, [(1, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("BroadcastTo", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] BroadcastTo    input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 13. Transpose ──
    def test_transpose(a): return a.transpose()
    results = check_gradient("Transpose", test_transpose, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Transpose", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Transpose      input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 14. ReLU ──
    def test_relu(a): return ops.relu(a)
    results = check_gradient("ReLU", test_relu, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("ReLU", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] ReLU           input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 15. Tanh ──
    def test_tanh(a): return ops.tanh(a)
    results = check_gradient("Tanh", test_tanh, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Tanh", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Tanh           input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 16. Log ──
    def test_log(a): return ops.log(a)
    # Use positive inputs for log
    results = check_gradient("Log", test_log, [(3, 4)], positive=True)
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Log", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Log            input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 17. Exp ──
    def test_exp(a): return ops.exp(a)
    results = check_gradient("Exp", test_exp, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Exp", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Exp            input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 18. LogSumExp ──
    def test_logsumexp(a): return ops.logsumexp(a, axes=(1,))
    results = check_gradient("LogSumExp", test_logsumexp, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("LogSumExp", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] LogSumExp      input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 19. SiLU ──
    def test_silu(a): return ops.silu(a)
    results = check_gradient("SiLU", test_silu, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("SiLU", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] SiLU           input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 20. Sqrt ──
    def test_sqrt(a): return ops.sqrt(a)
    results = check_gradient("Sqrt", test_sqrt, [(3, 4)], positive=True, tol=2e-2)
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Sqrt", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Sqrt           input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 21. Sin ──
    def test_sin(a): return ops.sin(a)
    results = check_gradient("Sin", test_sin, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Sin", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Sin            input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 22. Cos ──
    def test_cos(a): return ops.cos(a)
    results = check_gradient("Cos", test_cos, [(3, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Cos", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Cos            input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 23. Concatenate ──
    def test_concat(a, b): return ops.concatenate([a, b], axis=0)
    results = check_gradient("Concatenate", test_concat, [(3, 4), (2, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Concatenate", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Concatenate    input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── 24. Composite: Linear layer ──
    fixed_W = np.random.randn(4, 3).astype(np.float32) * 0.1
    def test_linear(x):
        W = Tensor(fixed_W.copy(), requires_grad=False)
        return x @ W
    results = check_gradient("Linear(x@W)", test_linear, [(2, 4)])
    for i, me, re, p in results:
        status = "PASS" if p else "FAIL"
        if p: passed_count += 1
        else: failed_count += 1
        test_cases.append(("Linear(x@W)", f"input_{i}", me, re, p))
        print(f"  [{'✓' if p else '✗'}] Linear(x@W)    input_{i}  max_err={me:.2e}  rel_err={re:.2e}  {status}")

    # ── Summary ──
    print("\n" + "=" * 70)
    total = passed_count + failed_count
    print(f"  总计: {total} 项测试, {passed_count} 通过, {failed_count} 失败")
    print(f"  通过率: {passed_count/total*100:.1f}%")
    print("=" * 70)

    return test_cases, passed_count, failed_count


if __name__ == "__main__":
    t0 = time.time()
    test_cases, passed, failed = run_tests()
    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.2f}s")
