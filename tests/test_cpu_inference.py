"""Minimal CPU backend test - just model creation, no weights."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python'))
import numpy as np
import uniti
from uniti.autograd import Tensor, no_grad
from uniti.nn.nn_qwen2 import Qwen2ForCausalLM
from uniti.backend_selection import cpu
import json

model_path = "/home/iclab/LLM_ndl/llaisys/models/DeepSeek-R1-Distill-Qwen-1.5B"
device = cpu()
print(f"Device: {device}", flush=True)

with open(os.path.join(model_path, "config.json")) as f:
    config = json.load(f)
print(f"Config loaded", flush=True)

# Only 1 layer for faster test
t0 = time.time()
print("Creating 1-layer model...", flush=True)
model = Qwen2ForCausalLM(
    vocab_size=config["vocab_size"],
    hidden_size=config["hidden_size"],
    num_hidden_layers=1,  # Only 1 layer!
    num_attention_heads=config["num_attention_heads"],
    num_key_value_heads=config["num_key_value_heads"],
    intermediate_size=config["intermediate_size"],
    rms_norm_eps=config.get("rms_norm_eps", 1e-6),
    max_position_embeddings=config.get("max_position_embeddings", 131072),
    rope_theta=config.get("rope_theta", 10000.0),
    tie_word_embeddings=config.get("tie_word_embeddings", False),
    device=device,
    dtype="float32",
)
print(f"  Model created in {time.time()-t0:.1f}s", flush=True)
print("Done!", flush=True)
