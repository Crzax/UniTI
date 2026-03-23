"""
MNIST Training Test — Two-layer FC network with manual SGD.

Downloads MNIST data automatically, then trains using nn_epoch / loss_err
from apps/simple_ml.py.

Usage:
    python tests/test_train_mnist.py                    # default: cpu_numpy
    python tests/test_train_mnist.py --device cpu        # 自研 C++ CPU 后端
    python tests/test_train_mnist.py --device cpu_numpy  # numpy 后端
    python tests/test_train_mnist.py --device cuda       # 自研 GPU 后端
"""
import sys
import os
import urllib.request
import time
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# 0.  Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))
sys.path.insert(0, PROJECT_ROOT)

import uniti as uti
from apps.simple_ml import parse_mnist, softmax_loss

# ---------------------------------------------------------------------------
# 1.  Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MNIST Two-layer FC Training")
parser.add_argument("--device", default="cpu_numpy",
                    choices=["cpu", "cpu_numpy", "cuda"],
                    help="Backend device: cpu (自研C++), cpu_numpy (numpy), cuda (自研GPU)")
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch", type=int, default=100)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 2.  Resolve device
# ---------------------------------------------------------------------------
DEVICE_MAP = {
    "cpu":       uti.cpu,
    "cpu_numpy": uti.cpu_numpy,
    "cuda":      uti.cuda,
}
device = DEVICE_MAP[args.device]()
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 3.  Download MNIST data
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "mnist")
os.makedirs(DATA_DIR, exist_ok=True)

MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
MIRROR_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_BASE = "http://yann.lecun.com/exdb/mnist/"

print("=" * 60)
print("Step 1: Download MNIST dataset")
print("=" * 60)

for fname in MNIST_FILES:
    dst = os.path.join(DATA_DIR, fname)
    if os.path.exists(dst):
        print(f"  [skip] {fname} already exists")
        continue
    for base_url in [MIRROR_BASE, MNIST_BASE]:
        url = base_url + fname
        print(f"  Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"  -> saved to {dst}")
            break
        except Exception as e:
            print(f"  [WARN] {base_url} failed: {e}")
    else:
        raise RuntimeError(f"Failed to download {fname}")

# ---------------------------------------------------------------------------
# 4.  Parse MNIST
# ---------------------------------------------------------------------------
print("\nStep 2: Parse MNIST data")
X_train, y_train = parse_mnist(
    os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"),
    os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"),
)
X_test, y_test = parse_mnist(
    os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"),
    os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"),
)
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 5.  Device-aware nn_epoch
#     (simple_ml.nn_epoch 内部 uti.Tensor(...) 使用默认设备，
#      这里封装一层以支持指定 device)
# ---------------------------------------------------------------------------
def nn_epoch(X, y, W1, W2, lr=0.1, batch=100, device=None):
    """与 simple_ml.nn_epoch 逻辑一致，但支持传入 device。"""
    iter_nums = (y.size + batch - 1) // batch
    for i in range(iter_nums):
        X_batch = uti.Tensor(X[i * batch : min((i + 1) * batch, y.size), :],
                             device=device)
        y_batch = y[i * batch : min((i + 1) * batch, y.size)]
        if i == iter_nums - 1:
            batch = y.size - i * batch
        y_one_hot = np.zeros((batch, y.max() + 1))
        y_one_hot[np.arange(batch), y_batch] = 1
        y_one_hot = uti.Tensor(y_one_hot, device=device)
        Z = uti.matmul(uti.relu(uti.matmul(X_batch, W1)), W2)
        loss = softmax_loss(Z, y_one_hot)
        loss.backward()
        W1 = uti.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data(),
                        device=device)
        W2 = uti.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data(),
                        device=device)
    return W1, W2


def compute_logits(X_np, W1, W2, device):
    """前向传播: h = relu(X @ W1) @ W2"""
    X_t = uti.Tensor(X_np, device=device)
    return uti.matmul(uti.relu(uti.matmul(X_t, W1)), W2)


def loss_err(h, y, device=None):
    """与 simple_ml.loss_err 逻辑一致，但支持指定 device。"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = uti.Tensor(y_one_hot, device=device)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)

# ---------------------------------------------------------------------------
# 6.  Train
# ---------------------------------------------------------------------------
HIDDEN = args.hidden
NUM_CLASSES = 10
N_EPOCHS = args.epochs
LR = args.lr
BATCH = args.batch

print(f"\nStep 3: Train two-layer FC network")
print(f"  device={args.device}, hidden={HIDDEN}, lr={LR}, batch={BATCH}, epochs={N_EPOCHS}")
print("-" * 60)

np.random.seed(0)
W1 = uti.Tensor(np.random.randn(784, HIDDEN).astype(np.float32) / HIDDEN ** 0.5,
                device=device)
W2 = uti.Tensor(np.random.randn(HIDDEN, NUM_CLASSES).astype(np.float32) / NUM_CLASSES ** 0.5,
                device=device)

t_start = time.time()
for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    W1, W2 = nn_epoch(X_train, y_train, W1, W2, lr=LR, batch=BATCH, device=device)
    dt = time.time() - t0

    train_h = compute_logits(X_train, W1, W2, device)
    train_loss, train_err = loss_err(train_h, y_train, device=device)
    test_h = compute_logits(X_test, W1, W2, device)
    test_loss, test_err = loss_err(test_h, y_test, device=device)
    print(f"  Epoch {epoch:2d}/{N_EPOCHS} | {dt:.1f}s | "
          f"train loss={train_loss.item():.4f} err={train_err.item():.4f} | "
          f"test  loss={test_loss.item():.4f}  err={test_err.item():.4f}")

total_time = time.time() - t_start

# ---------------------------------------------------------------------------
# 7.  Final report
# ---------------------------------------------------------------------------
final_h = compute_logits(X_test, W1, W2, device)
_, final_test_err = loss_err(final_h, y_test, device=device)
final_test_err = final_test_err.item()
final_test_acc = 1.0 - final_test_err

print(f"\n{'=' * 60}")
print(f"MNIST Training Complete!")
print(f"  Device     : {device}")
print(f"  Total time : {total_time:.1f}s")
print(f"  Test Acc   : {final_test_acc * 100:.2f}%")
print(f"  Test Error : {final_test_err * 100:.2f}%")
if final_test_acc > 0.95:
    print("  ✅ PASS — accuracy > 95%")
else:
    print("  ⚠️  accuracy < 95%, check implementation")
print(f"{'=' * 60}")
