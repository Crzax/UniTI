"""
DeepSeek-R1-Distill-Qwen-1.5B inference using UniTi framework.
Uses UniTi ops + no_grad() for training-inference unified pipeline.
Supports both CPU and GPU backends.
"""
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

import uniti
from uniti.autograd import Tensor, no_grad
from uniti.nn.nn_qwen2 import Qwen2ForCausalLM
from uniti.backend_selection import cuda, cpu_numpy, cpu


def get_device(device_name: str):
    """Get UniTi device by name."""
    if device_name == "cuda":
        dev = cuda()
        if not dev.enabled():
            raise RuntimeError("CUDA backend not available. Please compile with CUDA support.")
        return dev
    elif device_name == "cpu":
        return cpu()
    elif device_name == "cpu_numpy":
        return cpu_numpy()
    else:
        raise ValueError(f"Unknown device: {device_name}")


def load_safetensors(filepath):
    """Load weights from safetensors as float32 numpy arrays."""
    from safetensors import safe_open
    import torch
    tensors = {}
    with safe_open(filepath, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).to(torch.float32).numpy()
    return tensors


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_weights_into_model(model: Qwen2ForCausalLM, state_dict: dict, device=None):
    """Load HuggingFace weights into our UniTi Qwen2ForCausalLM model."""
    if device is None:
        device = model.device
    dtype = model.dtype
    num_layers = len(model.model.layers)

    def _set_param(param, np_arr):
        """Replace a Parameter's data with numpy array."""
        param.cached_data = Tensor._array_from_numpy(
            np_arr.astype(np.float32), device=device, dtype=dtype
        )

    def _set_linear(linear, weight_np, bias_np=None):
        """Load weight (out, in) -> transpose to (in, out) for UniTi Linear."""
        _set_param(linear.weight, weight_np.T)
        if bias_np is not None and linear.bias is not None:
            _set_param(linear.bias, bias_np.reshape(1, -1))

    # Embedding
    _set_param(model.model.embed_tokens.weight, state_dict["model.embed_tokens.weight"])

    # LM head
    if "lm_head.weight" in state_dict:
        _set_linear(model.lm_head, state_dict["lm_head.weight"])
    else:
        # Tied embeddings
        _set_linear(model.lm_head, state_dict["model.embed_tokens.weight"])

    # Final norm
    _set_param(model.model.norm.weight, state_dict["model.norm.weight"])

    # Layers
    for i in range(num_layers):
        p = f"model.layers.{i}"
        layer = model.model.layers[i]

        # Attention projections
        _set_linear(layer.self_attn.q_proj, state_dict[f"{p}.self_attn.q_proj.weight"],
                     state_dict[f"{p}.self_attn.q_proj.bias"])
        _set_linear(layer.self_attn.k_proj, state_dict[f"{p}.self_attn.k_proj.weight"],
                     state_dict[f"{p}.self_attn.k_proj.bias"])
        _set_linear(layer.self_attn.v_proj, state_dict[f"{p}.self_attn.v_proj.weight"],
                     state_dict[f"{p}.self_attn.v_proj.bias"])
        _set_linear(layer.self_attn.o_proj, state_dict[f"{p}.self_attn.o_proj.weight"])

        # MLP
        _set_linear(layer.mlp.gate_proj, state_dict[f"{p}.mlp.gate_proj.weight"])
        _set_linear(layer.mlp.up_proj, state_dict[f"{p}.mlp.up_proj.weight"])
        _set_linear(layer.mlp.down_proj, state_dict[f"{p}.mlp.down_proj.weight"])

        # Norms
        _set_param(layer.input_layernorm.weight, state_dict[f"{p}.input_layernorm.weight"])
        _set_param(layer.post_attention_layernorm.weight, state_dict[f"{p}.post_attention_layernorm.weight"])

    print(f"  Weights loaded into UniTi model ({num_layers} layers)")


def softmax_np(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def top_p_sampling(logits_np, temperature=0.6, top_p=0.95):
    logits_np = logits_np / temperature
    probs = softmax_np(logits_np.reshape(1, -1)).flatten()
    idx = np.argsort(-probs)
    cum = np.cumsum(probs[idx])
    mask = cum - probs[idx] > top_p
    probs_f = probs[idx].copy()
    probs_f[mask] = 0.0
    probs_f /= probs_f.sum()
    return int(np.random.choice(idx, p=probs_f))


def generate(model, input_ids, tokenizer, max_new_tokens=100,
             temperature=0.6, top_p=0.95, eos_token_id=151643, do_sample=True):
    model.reset_cache()
    model.eval()

    prompt_len = len(input_ids)
    max_cache_len = prompt_len + max_new_tokens
    model.init_cache(batch_size=1, max_cache_len=max_cache_len)

    ids_np = np.array([input_ids], dtype=np.float32)  # (1, prompt_len)

    with no_grad():
        # Prefill — use last_only=True to only compute lm_head for the last token
        t0 = time.time()
        ids_tensor = Tensor(ids_np, device=model.device, dtype=model.dtype, requires_grad=False)
        logits = model(ids_tensor, start_pos=0, last_only=True)
        logits_np = logits.realize_cached_data().numpy()
        last_logits = logits_np[0, -1, :]
        prefill_t = time.time() - t0
        print(f"  Prefill {prompt_len} tok in {prefill_t:.2f}s ({prompt_len / max(prefill_t, 1e-9):.1f} tok/s)")

    if do_sample and temperature > 0:
        next_tok = top_p_sampling(last_logits, temperature, top_p)
    else:
        next_tok = int(np.argmax(last_logits))

    generated = list(input_ids) + [next_tok]
    cur_pos = prompt_len
    print(tokenizer.decode([next_tok]), end="", flush=True)

    if next_tok == eos_token_id:
        print()
        return generated

    decode_times = []
    for step in range(1, max_new_tokens):
        t0 = time.time()
        with no_grad():
            tok_tensor = Tensor(
                np.array([[next_tok]], dtype=np.float32),
                device=model.device, dtype=model.dtype, requires_grad=False
            )
            logits = model(tok_tensor, start_pos=cur_pos, last_only=True)
            logits_np = logits.realize_cached_data().numpy()
            last_logits = logits_np[0, -1, :]
        cur_pos += 1
        dt = time.time() - t0
        decode_times.append(dt)

        if do_sample and temperature > 0:
            next_tok = top_p_sampling(last_logits, temperature, top_p)
        else:
            next_tok = int(np.argmax(last_logits))
        generated.append(next_tok)

        print(tokenizer.decode([next_tok]), end="", flush=True)
        if next_tok == eos_token_id:
            break

    print()
    if decode_times:
        avg = np.mean(decode_times)
        print(f"  Decode: {len(decode_times)} tok, avg {avg:.2f}s/tok ({1/avg:.2f} tok/s)")
    return generated


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/iclab/LLM_ndl/llaisys/models/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no_chat", action="store_true", help="Skip chat template, use raw prompt")
    parser.add_argument("--device", default="cpu_numpy", choices=["cuda", "cpu", "cpu_numpy"],
                        help="Device to run inference on (cuda/cpu/cpu_numpy)")
    args = parser.parse_args()

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load config
    print("Loading config...")
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    for k in ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size"]:
        print(f"  {k}: {config[k]}")

    # Create UniTi model with specified device
    print(f"Creating UniTi Qwen2ForCausalLM model on {args.device}...")
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

    # Load weights to device
    print("Loading weights...")
    t0 = time.time()
    state_dict = load_safetensors(os.path.join(args.model_path, "model.safetensors"))
    print(f"  {len(state_dict)} tensors in {time.time()-t0:.1f}s")
    load_weights_into_model(model, state_dict, device=device)
    del state_dict
    print(f"  Weights loaded to {args.device}")

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_path)

    # Tokenize (apply chat template by default)
    if args.no_chat:
        input_ids = tokenizer.encode(args.prompt)
    else:
        messages = [{"role": "user", "content": args.prompt}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(chat_text)
        print(f"Chat template applied: {repr(chat_text[:120])}...")

    print(f"\nPrompt: {args.prompt}")
    print(f"Tokens: {len(input_ids)}")
    print("-" * 50)

    # Generate
    t_start = time.time()
    output_ids = generate(
        model, input_ids, tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=config.get("eos_token_id", 151643),
        do_sample=not args.greedy,
    )
    total = time.time() - t_start

    gen_text = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True)
    n = len(output_ids) - len(input_ids)
    print(f"\n{'='*50}")
    print(f"Generated ({n} tokens in {total:.1f}s, {n/total:.2f} tok/s):")
    print(gen_text)


if __name__ == "__main__":
    main()
