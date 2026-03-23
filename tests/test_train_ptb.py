"""
PTB Language Model Training Test — RNN / LSTM / Transformer.

Downloads Penn Treebank data automatically, then trains using
train_ptb / evaluate_ptb from apps/simple_ml.py and
LanguageModel from apps/models.py.

Usage:
    python tests/test_train_ptb.py                              # default: cpu_numpy, lstm
    python tests/test_train_ptb.py --device cpu                  # 自研 C++ CPU 后端
    python tests/test_train_ptb.py --device cpu_numpy            # numpy 后端
    python tests/test_train_ptb.py --device cuda                 # 自研 GPU 后端
    python tests/test_train_ptb.py --device cuda --model rnn
    python tests/test_train_ptb.py --model transformer
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
import uniti.nn as nn
from apps.models import LanguageModel
from apps.simple_ml import train_ptb, evaluate_ptb

# ---------------------------------------------------------------------------
# 1.  Parse args (先解析，以便后续步骤使用 device)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="PTB Language Model Training")
parser.add_argument("--device", default="cpu_numpy",
                    choices=["cpu", "cpu_numpy", "cuda"],
                    help="Backend device: cpu (自研C++), cpu_numpy (numpy), cuda (自研GPU)")
parser.add_argument("--model", default="lstm", choices=["rnn", "lstm", "transformer"])
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--seq_len", type=int, default=40)
parser.add_argument("--lr", type=float, default=4.0)
parser.add_argument("--clip", type=float, default=0.25)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=16)
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
DTYPE = "float32"
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# 3.  Download PTB data
# ---------------------------------------------------------------------------
PTB_DIR = os.path.join(PROJECT_ROOT, "data", "ptb")
os.makedirs(PTB_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: Download Penn Treebank dataset")
print("=" * 60)

PTB_BASE = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
for fname in ["train.txt", "test.txt", "valid.txt"]:
    dst = os.path.join(PTB_DIR, fname)
    if os.path.exists(dst):
        print(f"  [skip] {fname} already exists")
        continue
    url = PTB_BASE + fname
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dst)
    print(f"  -> saved to {dst}")

# ---------------------------------------------------------------------------
# 4.  Build corpus & batchify
# ---------------------------------------------------------------------------
print(f"\nStep 2: Tokenize corpus")

corpus = uti.data.Corpus(PTB_DIR, max_lines=None)
vocab_size = len(corpus.dictionary)
print(f"  Vocabulary size : {vocab_size}")
print(f"  Train tokens    : {len(corpus.train)}")
print(f"  Test tokens     : {len(corpus.test)}")

train_data = uti.data.batchify(corpus.train, batch_size=args.batch_size,
                               device=device, dtype=DTYPE)
test_data = uti.data.batchify(corpus.test, batch_size=args.batch_size,
                              device=device, dtype=DTYPE)
print(f"  Train batches   : {train_data.shape}")
print(f"  Test  batches   : {test_data.shape}")

# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------
def main():
    # --- Model (直接用 apps/models.py 的 LanguageModel) ---
    print(f"\nStep 3: Create LanguageModel (seq_model={args.model})")
    model = LanguageModel(
        embedding_size=args.embedding_size,
        output_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        seq_model=args.model,
        device=device,
        dtype=DTYPE,
    )
    n_params = sum(np.prod(p.shape) for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # --- Train (直接调用 simple_ml.train_ptb) ---
    print(f"\nStep 4: Train ({args.epochs} epochs, lr={args.lr}, clip={args.clip})")
    print("-" * 60)

    t0 = time.time()
    train_acc, train_loss = train_ptb(
        model, train_data,
        seq_len=args.seq_len, n_epochs=args.epochs,
        optimizer=uti.optim.SGD, lr=args.lr,
        weight_decay=args.weight_decay, clip=args.clip,
        device=device, dtype=DTYPE,
    )
    train_time = time.time() - t0
    train_ppl = np.exp(train_loss.item())
    print(f"  Train done in {train_time:.1f}s — "
          f"loss={train_loss.item():.4f}, ppl={train_ppl:.1f}, acc={train_acc.item():.4f}")

    # --- Evaluate (直接调用 simple_ml.evaluate_ptb) ---
    print(f"\nStep 5: Evaluate on test set")
    test_acc, test_loss = evaluate_ptb(model, test_data,
                                       seq_len=args.seq_len,
                                       device=device, dtype=DTYPE)
    test_ppl = np.exp(test_loss.item())

    # --- Report ---
    print(f"\n{'=' * 60}")
    print(f"PTB Training Complete!")
    print(f"  Model       : {args.model.upper()}")
    print(f"  Device      : {device}")
    print(f"  Total time  : {train_time:.1f}s")
    print(f"  Test Loss   : {test_loss.item():.4f}")
    print(f"  Test PPL    : {test_ppl:.1f}")
    print(f"  Test Acc    : {test_acc.item() * 100:.2f}%")
    if test_ppl < 500:
        print(f"  ✅ PASS — perplexity < 500 (model is learning)")
    else:
        print(f"  ⚠️  perplexity >= 500, model may need more epochs or tuning")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
