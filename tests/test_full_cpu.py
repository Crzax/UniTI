"""Quick CPU inference test"""
import sys
import time
sys.path.insert(0, 'python')
import numpy as np
from uniti.autograd import Tensor, no_grad
from uniti.nn.nn_qwen2 import Qwen2ForCausalLM
from uniti.backend_selection import cpu_numpy
import json

device = cpu_numpy()
print(f'Device: {device}')

# Load config
model_path = "/home/iclab/LLM_ndl/llaisys/models/DeepSeek-R1-Distill-Qwen-1.5B"
with open(f"{model_path}/config.json") as f:
    config = json.load(f)
print(f"Creating model with {config['num_hidden_layers']} layers...")

t0 = time.time()
model = Qwen2ForCausalLM(
    vocab_size=config["vocab_size"],
    hidden_size=config["hidden_size"],
    num_hidden_layers=config["num_hidden_layers"],
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
print(f"Model created in {time.time()-t0:.1f}s")

# Load weights
print("Loading weights...")
from safetensors import safe_open
import torch
t0 = time.time()
state_dict = {}
with safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key).to(torch.float32).numpy()
print(f"Weights loaded in {time.time()-t0:.1f}s")

# Simple test: just a forward pass
model.eval()
model.init_cache(batch_size=1, max_cache_len=10)

input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
print(f"\nTesting forward with shape {input_ids.shape}...")

t0 = time.time()
with no_grad():
    x = Tensor(input_ids, device=device, dtype="float32", requires_grad=False)
    logits = model(x, start_pos=0, last_only=True)
print(f"Forward pass: {time.time()-t0:.2f}s")
print(f"Output shape: {logits.shape}")
