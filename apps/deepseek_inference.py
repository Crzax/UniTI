"""
DeepSeek-R1-Distill-Qwen-1.5B inference using UniTi framework.
Uses UniTi ops + no_grad() for training-inference unified pipeline.
Supports both CPU and GPU backends.

External dependencies: only numpy (no safetensors / torch / transformers needed).
Weight loading and tokenization are handled by UniTi's own implementations.
"""
import sys
import os
import json
import time
import codecs
import gc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

import uniti
from uniti.autograd import Tensor, no_grad
from uniti.nn.nn_qwen2 import Qwen2ForCausalLM
from uniti.backend_selection import cuda, cpu_numpy, cpu, metal
from uniti.safetensors_loader import load_safetensors, load_safetensors_sharded
from uniti.tokenizer import UniTITokenizer


def get_device(device_name: str):
    """Get UniTi device by name."""
    if device_name == "cuda":
        dev = cuda()
        if not dev.enabled():
            raise RuntimeError("CUDA backend not available. Please compile with CUDA support.")
        return dev
    elif device_name == "metal":
        dev = metal()
        if not dev.enabled():
            raise RuntimeError("Metal backend not available. Requires macOS + Apple Silicon + metalcompute.")
        return dev
    elif device_name == "cpu":
        return cpu()
    elif device_name == "cpu_numpy":
        return cpu_numpy()
    else:
        raise ValueError(f"Unknown device: {device_name}")


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
    model.model.embed_tokens._invalidate_cache()  # clear numpy cache after weight update

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


def _release_cache(model, use_paged_cache, cache_mgr=None):
    """Proactively release KV cache memory after generation completes.
    
    Without this, KV cache blocks remain allocated in GPU/CPU memory until
    the next generate() call invokes model.reset_cache(). This can be
    problematic in interactive settings where idle time between calls is long.
    
    For Paged Attention: frees all sequence page-table entries (logical release),
    which allows the blocks to be reused immediately.
    
    For Contiguous cache: sets cache NDArrays to None so Python GC can
    reclaim the underlying memory (cudaFree / free).
    
    Calls gc.collect() to ensure prompt reclamation of cyclic references.
    """
    if use_paged_cache and cache_mgr is not None:
        freed = cache_mgr.free_all_sequences()
        if freed > 0:
            print(f"  [Memory] Released {freed} paged blocks "
                  f"(~{cache_mgr.memory_footprint_bytes / 1024 / 1024:.1f} MB pool)")
    # Reset model-level cache references (sets NDArray refs to None)
    model.reset_cache()
    # Force GC to reclaim any cyclic references promptly
    gc.collect()


def generate(model, input_ids, tokenizer, max_new_tokens=100,
             temperature=0.6, top_p=0.95, eos_token_id=151643, do_sample=True,
             use_paged_cache=False, paged_block_size=16, paged_max_blocks=256):
    model.reset_cache()
    model.eval()

    prompt_len = len(input_ids)
    max_cache_len = prompt_len + max_new_tokens

    if use_paged_cache:
        # Paged Attention mode: allocate blocks on-demand
        cache_mgr = model.init_paged_cache(
            block_size=paged_block_size,
            max_num_blocks=paged_max_blocks,
            seq_id=0,
            initial_len=0,  # blocks allocated dynamically during forward
        )
        print(f"  [Paged Attention] block_size={paged_block_size}, "
              f"max_blocks={paged_max_blocks}, "
              f"pool_size={paged_max_blocks * paged_block_size} tokens")
    else:
        # Contiguous cache mode (legacy)
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

    if use_paged_cache:
        stats = cache_mgr.get_cache_stats()
        print(f"  [Paged] After prefill: {stats['num_used_blocks']}/{stats['max_num_blocks']} blocks "
              f"({stats['utilization']:.1f}% used)")

    if do_sample and temperature > 0:
        next_tok = top_p_sampling(last_logits, temperature, top_p)
    else:
        next_tok = int(np.argmax(last_logits))

    generated = list(input_ids) + [next_tok]
    cur_pos = prompt_len

    # Incremental UTF-8 decoder for streaming output.
    # Byte-level BPE may split multi-byte chars (emoji) across tokens.
    # The incremental decoder buffers incomplete byte sequences internally
    # and only emits complete characters.
    utf8_decoder = codecs.getincrementaldecoder('utf-8')('ignore')

    def _stream_token(tok_id):
        """Feed one token's raw bytes into the incremental decoder and print."""
        raw = tokenizer.token_to_bytes(tok_id, skip_special_tokens=False)
        text = utf8_decoder.decode(raw, False)
        if text:
            print(text, end="", flush=True)

    _stream_token(next_tok)

    if next_tok == eos_token_id:
        # Flush any buffered bytes
        tail = utf8_decoder.decode(b'', True)
        if tail:
            print(tail, end="", flush=True)
        print()
        # Proactively release KV cache after generation completes
        _release_cache(model, use_paged_cache, cache_mgr if use_paged_cache else None)
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

        _stream_token(next_tok)

        if next_tok == eos_token_id:
            break

    # Flush any remaining bytes in the incremental decoder
    tail = utf8_decoder.decode(b'', True)
    if tail:
        print(tail, end="", flush=True)
    print()
    if decode_times:
        avg = np.mean(decode_times)
        print(f"  Decode: {len(decode_times)} tok, avg {avg:.2f}s/tok ({1/avg:.2f} tok/s)")
    
    if use_paged_cache:
        stats = cache_mgr.get_cache_stats()
        print(f"  [Paged] Final: {stats['num_used_blocks']}/{stats['max_num_blocks']} blocks "
              f"({stats['utilization']:.1f}% used)")
    
    # Proactively release KV cache after generation completes
    _release_cache(model, use_paged_cache, cache_mgr if use_paged_cache else None)
    return generated


def generate_batch(model, batch_input_ids, tokenizer, max_new_tokens=100,
                   temperature=0.6, top_p=0.95, eos_token_id=151643, do_sample=True,
                   paged_block_size=16, paged_max_blocks=512):
    """Batch inference using Paged Attention.
    
    Each sequence in the batch can have a different prompt length.
    Sequences that reach EOS stop generating but stay in the batch
    (padded) until all sequences are done.
    
    Args:
        model: Qwen2ForCausalLM model.
        batch_input_ids: List of token ID lists (one per sequence).
        tokenizer: UniTITokenizer instance.
        max_new_tokens: Maximum number of tokens to generate per sequence.
        temperature: Sampling temperature.
        top_p: Top-p sampling threshold.
        eos_token_id: End-of-sequence token ID.
        do_sample: If True, use top-p sampling; else greedy.
        paged_block_size: Block size for Paged Attention.
        paged_max_blocks: Maximum number of physical blocks.
        
    Returns:
        List of generated token ID lists (one per sequence, including prompt).
    """
    model.reset_cache()
    model.eval()
    
    batch_size = len(batch_input_ids)
    prompt_lens = [len(ids) for ids in batch_input_ids]
    
    print(f"  [Batch Paged Attention] batch_size={batch_size}, "
          f"block_size={paged_block_size}, max_blocks={paged_max_blocks}")
    print(f"  Prompt lengths: {prompt_lens}")
    
    # Initialize paged cache for all sequences
    seq_ids = list(range(batch_size))
    cache_mgr = model.init_paged_cache(
        block_size=paged_block_size,
        max_num_blocks=paged_max_blocks,
        seq_id=seq_ids,
        initial_len=0,
    )
    
    # === Phase 1: Prefill each sequence individually ===
    # (Different prompt lengths require separate prefill passes)
    t0 = time.time()
    first_logits = []
    
    with no_grad():
        for b in range(batch_size):
            prompt = batch_input_ids[b]
            pl = len(prompt)
            ids_np = np.array([prompt], dtype=np.float32)  # (1, prompt_len)
            
            # Temporarily set the attention layers to only use this sequence's ID
            for layer in model.model.layers:
                layer.self_attn._paged_seq_ids = [seq_ids[b]]
                layer.self_attn._paged_seq_id = seq_ids[b]
            
            ids_tensor = Tensor(ids_np, device=model.device, dtype=model.dtype, requires_grad=False)
            logits = model(ids_tensor, start_pos=0, last_only=True)
            logits_np = logits.realize_cached_data().numpy()
            first_logits.append(logits_np[0, -1, :])
    
    prefill_t = time.time() - t0
    total_prefill_toks = sum(prompt_lens)
    print(f"  Prefill {batch_size} sequences ({total_prefill_toks} total tokens) "
          f"in {prefill_t:.2f}s ({total_prefill_toks / max(prefill_t, 1e-9):.1f} tok/s)")
    
    # Restore the full seq_ids list for batch decode
    for layer in model.model.layers:
        layer.self_attn._paged_seq_ids = seq_ids
        layer.self_attn._paged_seq_id = seq_ids[0]
    
    # Sample first tokens
    next_toks = []
    for b in range(batch_size):
        if do_sample and temperature > 0:
            tok = top_p_sampling(first_logits[b], temperature, top_p)
        else:
            tok = int(np.argmax(first_logits[b]))
        next_toks.append(tok)
    
    # Initialize generated sequences and tracking
    generated = [list(batch_input_ids[b]) + [next_toks[b]] for b in range(batch_size)]
    cur_pos = list(prompt_lens)  # per-sequence position
    finished = [next_toks[b] == eos_token_id for b in range(batch_size)]
    
    stats = cache_mgr.get_cache_stats()
    print(f"  [Paged] After prefill: {stats['num_used_blocks']}/{stats['max_num_blocks']} blocks "
          f"({stats['utilization']:.1f}% used)")
    
    # === Phase 2: Batch decode ===
    decode_times = []
    
    for step in range(1, max_new_tokens):
        if all(finished):
            break
        
        t0 = time.time()
        with no_grad():
            # Build batch input: (batch_size, 1)
            tok_np = np.array([[next_toks[b]] for b in range(batch_size)], dtype=np.float32)
            tok_tensor = Tensor(
                tok_np, device=model.device, dtype=model.dtype, requires_grad=False
            )
            logits = model(tok_tensor, start_pos=cur_pos, last_only=True)
            logits_np = logits.realize_cached_data().numpy()
        
        dt = time.time() - t0
        decode_times.append(dt)
        
        # Sample next tokens
        for b in range(batch_size):
            if finished[b]:
                continue
            cur_pos[b] += 1
            last_logits = logits_np[b, -1, :]
            if do_sample and temperature > 0:
                next_tok = top_p_sampling(last_logits, temperature, top_p)
            else:
                next_tok = int(np.argmax(last_logits))
            next_toks[b] = next_tok
            generated[b].append(next_tok)
            if next_tok == eos_token_id:
                finished[b] = True
                print(f"  [EOS] Seq {b} finished at step {step}")
    
    # Print stats
    if decode_times:
        avg = np.mean(decode_times)
        print(f"  Decode: {len(decode_times)} steps, avg {avg:.3f}s/step "
              f"({batch_size/avg:.2f} effective tok/s for batch_size={batch_size})")
    
    stats = cache_mgr.get_cache_stats()
    print(f"  [Paged] Final: {stats['num_used_blocks']}/{stats['max_num_blocks']} blocks "
          f"({stats['utilization']:.1f}% used)")
    
    for b in range(batch_size):
        gen_len = len(generated[b]) - prompt_lens[b]
        print(f"  Seq {b}: prompt={prompt_lens[b]}, generated={gen_len}, "
              f"finished={'yes' if finished[b] else 'no'}")
    
    # Proactively release all remaining KV cache after batch generation completes
    _release_cache(model, True, cache_mgr)
    return generated


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="")
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no_chat", action="store_true", help="Skip chat template, use raw prompt")
    parser.add_argument("--device", default="cpu_numpy", choices=["cuda", "cpu", "cpu_numpy", "metal"],
                        help="Device to run inference on (cuda/cpu/cpu_numpy/metal)")
    parser.add_argument("--paged", action="store_true",
                        help="Use Paged Attention for KV cache (vLLM-style block memory management)")
    parser.add_argument("--block_size", type=int, default=16,
                        help="Block size for Paged Attention (tokens per page)")
    parser.add_argument("--max_blocks", type=int, default=256,
                        help="Maximum number of physical blocks for Paged Attention")
    parser.add_argument("--batch", action="store_true",
                        help="Enable batch inference mode (uses Paged Attention)")
    parser.add_argument("--batch_prompts", nargs="+", default=None,
                        help="Multiple prompts for batch inference (one per sequence)")
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

    # Load weights to device (native loader, no safetensors/torch dependency)
    print("Loading weights (UniTi native safetensors loader)...")
    t0 = time.time()
    state_dict = load_safetensors_sharded(args.model_path)
    print(f"  {len(state_dict)} tensors in {time.time()-t0:.1f}s")
    load_weights_into_model(model, state_dict, device=device)
    del state_dict
    print(f"  Weights loaded to {args.device}")

    # Tokenizer (UniTi native BPE tokenizer, no transformers dependency)
    print("Loading tokenizer (UniTi native BPE tokenizer)...")
    tokenizer = UniTITokenizer.from_pretrained(args.model_path)

    eos_token_id = config.get("eos_token_id", tokenizer.eos_token_id or 151643)

    if args.batch:
        # === Batch Inference Mode ===
        prompts = args.batch_prompts or [
            args.prompt,
            "What is deep learning?",
            "Tell me a joke.",
        ]
        print(f"\n=== Batch Inference Mode ({len(prompts)} sequences) ===")

        # Tokenize all prompts
        batch_input_ids = []
        for i, prompt in enumerate(prompts):
            if args.no_chat:
                ids = tokenizer.encode(prompt)
            else:
                messages = [{"role": "user", "content": prompt}]
                chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ids = tokenizer.encode(chat_text)
            batch_input_ids.append(ids)
            print(f"  Prompt {i}: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\" ({len(ids)} tokens)")

        print("-" * 50)
        t_start = time.time()
        batch_output_ids = generate_batch(
            model, batch_input_ids, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
            do_sample=not args.greedy,
            paged_block_size=args.block_size,
            paged_max_blocks=args.max_blocks,
        )
        total = time.time() - t_start

        print(f"\n{'='*50}")
        total_gen = 0
        for i in range(len(prompts)):
            gen_ids = batch_output_ids[i][len(batch_input_ids[i]):]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            n = len(gen_ids)
            total_gen += n
            print(f"\n--- Sequence {i} ({n} tokens) ---")
            print(f"Prompt: {prompts[i]}")
            print(f"Response: {gen_text}")
        print(f"\n{'='*50}")
        print(f"Total: {total_gen} tokens from {len(prompts)} sequences in {total:.1f}s "
              f"({total_gen/total:.2f} tok/s effective throughput)")

    else:
        # === Single Sequence Mode ===
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

        if args.paged:
            print(f"KV Cache mode: Paged Attention (block_size={args.block_size}, max_blocks={args.max_blocks})")
        else:
            print(f"KV Cache mode: Contiguous (pre-allocated)")

        t_start = time.time()
        output_ids = generate(
            model, input_ids, tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
            do_sample=not args.greedy,
            use_paged_cache=args.paged,
            paged_block_size=args.block_size,
            paged_max_blocks=args.max_blocks,
        )
        total = time.time() - t_start

        gen_text = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True)
        n = len(output_ids) - len(input_ids)
        print(f"\n{'='*50}")
        print(f"Generated ({n} tokens in {total:.1f}s, {n/total:.2f} tok/s):")
        print(gen_text)


if __name__ == "__main__":
    main()
