"""Quick CPU inference test - without loading weights"""
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
model_path = ""
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

# Check cache status
model.eval()
model.init_cache(batch_size=1, max_cache_len=10)
print("Cache initialized")

# Check attention layers
attn = model.model.layers[0].self_attn
print(f"Layer 0 attention:")
print(f"  _k_cache type: {type(attn._k_cache)}")
print(f"  _cache_np_k type: {type(attn._cache_np_k)}")
