"""
CIFAR-10 Training Test — ResNet9 with ConvBN layers.

Downloads CIFAR-10 data automatically, then trains ResNet9 using
the train_cifar10 / evaluate_cifar10 functions from apps/simple_ml.py.

Usage:
    python tests/test_train_cifar10.py                        # default: cpu_numpy
    python tests/test_train_cifar10.py --device cpu            # 自研 C++ CPU 后端
    python tests/test_train_cifar10.py --device cpu_numpy      # numpy 后端
    python tests/test_train_cifar10.py --device cuda           # 自研 GPU 后端
    python tests/test_train_cifar10.py --device cuda --epochs 10 --lr 0.001
"""
import sys
import os
import urllib.request
import tarfile
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
import uniti.nn as nn
from apps.models import ResNet9
from apps.simple_ml import train_cifar10, evaluate_cifar10

# ---------------------------------------------------------------------------
# 1.  Download CIFAR-10 data
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CIFAR_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")
CIFAR_TAR = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")

print("=" * 60)
print("Step 1: Download CIFAR-10 dataset")
print("=" * 60)

if not os.path.isdir(CIFAR_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CIFAR_TAR):
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, CIFAR_TAR)
        print(f"  -> saved to {CIFAR_TAR}")
    print(f"  Extracting {CIFAR_TAR} ...")
    with tarfile.open(CIFAR_TAR, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    print(f"  -> extracted to {CIFAR_DIR}")
else:
    print(f"  [skip] {CIFAR_DIR} already exists")

# ---------------------------------------------------------------------------
# 2.  Main
# ---------------------------------------------------------------------------
DEVICE_MAP = {
    "cpu":       uti.cpu,
    "cpu_numpy": uti.cpu_numpy,
    "cuda":      uti.cuda,
}


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet9 Training Test")
    parser.add_argument("--device", default="cpu_numpy",
                        choices=["cpu", "cpu_numpy", "cuda"],
                        help="Backend device: cpu (自研C++), cpu_numpy (numpy), cuda (自研GPU)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = DEVICE_MAP[args.device]()
    print(f"\nUsing device: {device}")

    # --- Dataset & DataLoader ---
    print(f"\nStep 2: Load CIFAR-10 dataset")
    train_dataset = uti.data.CIFAR10Dataset(CIFAR_DIR, train=True)
    test_dataset = uti.data.CIFAR10Dataset(CIFAR_DIR, train=False)
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = uti.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       shuffle=True, device=device, dtype="float32")
    test_loader = uti.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                      shuffle=False, device=device, dtype="float32")

    # --- Model ---
    print(f"\nStep 3: Create ResNet9 model (device={device})")
    model = ResNet9(device=device, dtype="float32")
    n_params = sum(np.prod(p.shape) for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # --- Train (直接调用 simple_ml.train_cifar10) ---
    print(f"\nStep 4: Train (epochs={args.epochs}, lr={args.lr})")
    print("-" * 60)

    t0 = time.time()
    train_acc, train_loss = train_cifar10(
        model, train_loader,
        n_epochs=args.epochs, optimizer=uti.optim.Adam,
        lr=args.lr, weight_decay=args.weight_decay
    )
    train_time = time.time() - t0
    print(f"  Train done in {train_time:.1f}s — acc={train_acc:.4f}, loss={float(train_loss):.4f}")

    # --- Evaluate (直接调用 simple_ml.evaluate_cifar10) ---
    print(f"\nStep 5: Evaluate on test set")
    test_acc, test_loss = evaluate_cifar10(model, test_loader)

    # --- Report ---
    print(f"\n{'=' * 60}")
    print(f"CIFAR-10 Training Complete!")
    print(f"  Model      : ResNet9")
    print(f"  Device     : {device}")
    print(f"  Total time : {train_time:.1f}s")
    print(f"  Train Acc  : {train_acc * 100:.2f}%")
    print(f"  Test  Acc  : {test_acc * 100:.2f}%")
    print(f"  Test  Loss : {float(test_loss):.4f}")
    if test_acc > 0.60:
        print(f"  ✅ PASS — ResNet9 accuracy > 60%")
    elif test_acc > 0.40:
        print(f"  ⚠️  Accuracy 40%-60%, model is learning but may need more epochs")
    else:
        print(f"  ❌ FAIL — accuracy < 40%, check Conv/ConvBN implementation")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
