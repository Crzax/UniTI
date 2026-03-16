"""
Benchmark: Paged Attention vs Contiguous KV Cache — Full Model Inference
=========================================================================

Performance comparison on real DeepSeek-R1-Distill-Qwen-1.5B model.
Measures end-to-end inference metrics under identical conditions.

Metrics:
  1. Prefill Latency (ms)         — Time to process the prompt
  2. Prefill Throughput (tok/s)    — Tokens processed per second during prefill
  3. Decode Latency (ms/tok)      — Average time per generated token
  4. Decode Throughput (tok/s)     — Tokens generated per second
  5. Time To First Token (ms)     — Time from start to first generated token (TTFT)
  6. Per-step Latency Trend       — How decode latency changes as seq grows
  7. Total Generation Time (s)    — Wall clock time for the full generation
  8. GPU Memory (MB)              — Peak/init/final GPU memory (CUDA only)
  9. KV Cache Memory Efficiency   — Actual vs reserved memory

Usage:
  # CUDA benchmark (recommended):
  python3.10 tests/bench_paged_vs_contiguous.py --device cuda --max_new_tokens 100

  # CPU benchmark:
  python3.10 tests/bench_paged_vs_contiguous.py --device cpu_numpy --max_new_tokens 30

  # Quick test:
  python3.10 tests/bench_paged_vs_contiguous.py --device cuda --max_new_tokens 50 --greedy
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from datetime import datetime
import functools

# Force unbuffered output for long-running benchmarks
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))

import uniti
from uniti.autograd import Tensor, no_grad
from uniti.nn.nn_qwen2 import Qwen2ForCausalLM
from uniti.backend_selection import cuda, cpu_numpy, cpu


# ─────────────────────────────────────────────────────────────
#  GPU Memory Utilities
# ─────────────────────────────────────────────────────────────

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return None


def sync_cuda_device():
    """Synchronize CUDA device to get accurate timing."""
    try:
        import ctypes
        libcudart = ctypes.CDLL('libcudart.so')
        libcudart.cudaDeviceSynchronize()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
#  Model Loading (shared across benchmarks)
# ─────────────────────────────────────────────────────────────

def get_device(device_name):
    if device_name == "cuda":
        dev = cuda()
        if not dev.enabled():
            raise RuntimeError("CUDA not available")
        return dev
    elif device_name == "cpu":
        return cpu()
    elif device_name == "cpu_numpy":
        return cpu_numpy()
    else:
        raise ValueError(f"Unknown device: {device_name}")


def load_model_and_tokenizer(model_path, device):
    """Load model and tokenizer once, return (model, tokenizer, config)."""
    from safetensors import safe_open
    import torch

    # Config
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)

    # Model
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

    # Weights
    state_dict = {}
    with safe_open(os.path.join(model_path, "model.safetensors"), framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key).to(torch.float32).numpy()

    # Load weights into model (inline to avoid importing deepseek_inference)
    dtype = model.dtype
    num_layers = len(model.model.layers)

    def _set_param(param, np_arr):
        param.cached_data = Tensor._array_from_numpy(
            np_arr.astype(np.float32), device=device, dtype=dtype
        )

    def _set_linear(linear, weight_np, bias_np=None):
        _set_param(linear.weight, weight_np.T)
        if bias_np is not None and linear.bias is not None:
            _set_param(linear.bias, bias_np.reshape(1, -1))

    _set_param(model.model.embed_tokens.weight, state_dict["model.embed_tokens.weight"])
    model.model.embed_tokens._invalidate_cache()

    if "lm_head.weight" in state_dict:
        _set_linear(model.lm_head, state_dict["lm_head.weight"])
    else:
        _set_linear(model.lm_head, state_dict["model.embed_tokens.weight"])
    _set_param(model.model.norm.weight, state_dict["model.norm.weight"])

    for i in range(num_layers):
        p = f"model.layers.{i}"
        layer = model.model.layers[i]
        _set_linear(layer.self_attn.q_proj, state_dict[f"{p}.self_attn.q_proj.weight"],
                     state_dict[f"{p}.self_attn.q_proj.bias"])
        _set_linear(layer.self_attn.k_proj, state_dict[f"{p}.self_attn.k_proj.weight"],
                     state_dict[f"{p}.self_attn.k_proj.bias"])
        _set_linear(layer.self_attn.v_proj, state_dict[f"{p}.self_attn.v_proj.weight"],
                     state_dict[f"{p}.self_attn.v_proj.bias"])
        _set_linear(layer.self_attn.o_proj, state_dict[f"{p}.self_attn.o_proj.weight"])
        _set_linear(layer.mlp.gate_proj, state_dict[f"{p}.mlp.gate_proj.weight"])
        _set_linear(layer.mlp.up_proj, state_dict[f"{p}.mlp.up_proj.weight"])
        _set_linear(layer.mlp.down_proj, state_dict[f"{p}.mlp.down_proj.weight"])
        _set_param(layer.input_layernorm.weight, state_dict[f"{p}.input_layernorm.weight"])
        _set_param(layer.post_attention_layernorm.weight, state_dict[f"{p}.post_attention_layernorm.weight"])

    del state_dict

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return model, tokenizer, config


# ─────────────────────────────────────────────────────────────
#  Sampling
# ─────────────────────────────────────────────────────────────

def softmax_np(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def greedy_sample(logits_np):
    return int(np.argmax(logits_np))


# ─────────────────────────────────────────────────────────────
#  Benchmark Runner
# ─────────────────────────────────────────────────────────────

def run_benchmark(model, input_ids, max_new_tokens, eos_token_id,
                  use_paged, block_size=16, max_blocks=256, is_cuda=False):
    """
    Run a single benchmark (paged or contiguous) and collect metrics.

    Returns a dict with all measured metrics.
    """
    model.reset_cache()
    model.eval()

    prompt_len = len(input_ids)
    max_cache_len = prompt_len + max_new_tokens

    # ── GPU memory: before cache init ──
    if is_cuda:
        sync_cuda_device()
    mem_before_cache = get_gpu_memory_mb() if is_cuda else None

    # ── Initialize cache ──
    cache_mgr = None
    if use_paged:
        cache_mgr = model.init_paged_cache(
            block_size=block_size,
            max_num_blocks=max_blocks,
            seq_id=0,
            initial_len=0,
        )
    else:
        model.init_cache(batch_size=1, max_cache_len=max_cache_len)

    if is_cuda:
        sync_cuda_device()
    mem_after_cache = get_gpu_memory_mb() if is_cuda else None

    ids_np = np.array([input_ids], dtype=np.float32)

    # ── Prefill ──
    if is_cuda:
        sync_cuda_device()

    t_start_total = time.perf_counter()
    t_prefill_start = time.perf_counter()
    with no_grad():
        ids_tensor = Tensor(ids_np, device=model.device, dtype=model.dtype, requires_grad=False)
        logits = model(ids_tensor, start_pos=0, last_only=True)
        logits_np = logits.realize_cached_data().numpy()
        last_logits = logits_np[0, -1, :]

    if is_cuda:
        sync_cuda_device()
    t_prefill_end = time.perf_counter()
    prefill_time_s = t_prefill_end - t_prefill_start

    mem_after_prefill = get_gpu_memory_mb() if is_cuda else None

    next_tok = greedy_sample(last_logits)
    ttft = time.perf_counter() - t_start_total  # Time To First Token

    generated_tokens = [next_tok]
    cur_pos = prompt_len

    if next_tok == eos_token_id:
        t_total = time.perf_counter() - t_start_total
        return _build_result(
            mode="paged" if use_paged else "contiguous",
            prompt_len=prompt_len,
            num_generated=1,
            prefill_time_s=prefill_time_s,
            ttft_s=ttft,
            decode_times=[],
            total_time_s=t_total,
            mem_before_cache=mem_before_cache,
            mem_after_cache=mem_after_cache,
            mem_after_prefill=mem_after_prefill,
            mem_peak=mem_after_prefill,
            mem_final=mem_after_prefill,
            cache_mgr=cache_mgr,
            block_size=block_size,
            max_blocks=max_blocks,
            max_cache_len=max_cache_len,
            config_info=None,
        )

    # ── Decode ──
    decode_times = []
    mem_peak = mem_after_prefill or 0

    for step in range(1, max_new_tokens):
        if is_cuda:
            sync_cuda_device()
        t0 = time.perf_counter()

        with no_grad():
            tok_tensor = Tensor(
                np.array([[next_tok]], dtype=np.float32),
                device=model.device, dtype=model.dtype, requires_grad=False
            )
            logits = model(tok_tensor, start_pos=cur_pos, last_only=True)
            logits_np = logits.realize_cached_data().numpy()
            last_logits = logits_np[0, -1, :]

        if is_cuda:
            sync_cuda_device()
        dt = time.perf_counter() - t0
        decode_times.append(dt)

        cur_pos += 1
        next_tok = greedy_sample(last_logits)
        generated_tokens.append(next_tok)

        # Sample GPU memory at intervals
        if is_cuda and (step % 10 == 0 or step <= 5):
            mem_now = get_gpu_memory_mb()
            if mem_now and mem_now > mem_peak:
                mem_peak = mem_now

        if next_tok == eos_token_id:
            break

    if is_cuda:
        sync_cuda_device()
    mem_final = get_gpu_memory_mb() if is_cuda else None
    if mem_final and mem_final > mem_peak:
        mem_peak = mem_final

    t_total = time.perf_counter() - t_start_total

    return _build_result(
        mode="paged" if use_paged else "contiguous",
        prompt_len=prompt_len,
        num_generated=len(generated_tokens),
        prefill_time_s=prefill_time_s,
        ttft_s=ttft,
        decode_times=decode_times,
        total_time_s=t_total,
        mem_before_cache=mem_before_cache,
        mem_after_cache=mem_after_cache,
        mem_after_prefill=mem_after_prefill,
        mem_peak=mem_peak,
        mem_final=mem_final,
        cache_mgr=cache_mgr,
        block_size=block_size,
        max_blocks=max_blocks,
        max_cache_len=max_cache_len,
        config_info=None,
    )


def _build_result(mode, prompt_len, num_generated, prefill_time_s, ttft_s,
                  decode_times, total_time_s, mem_before_cache, mem_after_cache,
                  mem_after_prefill, mem_peak, mem_final, cache_mgr,
                  block_size, max_blocks, max_cache_len, config_info):
    """Assemble all metrics into a result dict."""
    decode_times_ms = [dt * 1000 for dt in decode_times]

    # Split decode latency into segments to show trend
    n = len(decode_times_ms)
    segments = {}
    if n >= 10:
        seg_size = n // 5
        for i, label in enumerate(["first_20%", "20-40%", "40-60%", "60-80%", "last_20%"]):
            start = i * seg_size
            end = (i + 1) * seg_size if i < 4 else n
            segments[label] = np.mean(decode_times_ms[start:end])
    elif n > 0:
        segments["first_half"] = np.mean(decode_times_ms[:max(1, n//2)])
        segments["second_half"] = np.mean(decode_times_ms[max(1, n//2):])

    result = {
        "mode": mode,
        "prompt_len": prompt_len,
        "num_generated": num_generated,
        "total_tokens": prompt_len + num_generated,

        # Latency
        "prefill_time_ms": prefill_time_s * 1000,
        "prefill_throughput_tok_s": prompt_len / max(prefill_time_s, 1e-9),
        "ttft_ms": ttft_s * 1000,
        "total_time_s": total_time_s,

        # Decode
        "decode_avg_ms": np.mean(decode_times_ms) if decode_times_ms else 0,
        "decode_median_ms": np.median(decode_times_ms) if decode_times_ms else 0,
        "decode_p90_ms": np.percentile(decode_times_ms, 90) if decode_times_ms else 0,
        "decode_p99_ms": np.percentile(decode_times_ms, 99) if decode_times_ms else 0,
        "decode_min_ms": np.min(decode_times_ms) if decode_times_ms else 0,
        "decode_max_ms": np.max(decode_times_ms) if decode_times_ms else 0,
        "decode_throughput_tok_s": 1000.0 / np.mean(decode_times_ms) if decode_times_ms and np.mean(decode_times_ms) > 0 else 0,
        "decode_std_ms": np.std(decode_times_ms) if decode_times_ms else 0,

        # Per-step latency (for trend analysis)
        "decode_times_ms": decode_times_ms,
        "decode_segments": segments,

        # Overall throughput (including prefill)
        "overall_throughput_tok_s": num_generated / max(total_time_s, 1e-9),

        # GPU Memory
        "mem_before_cache_mb": mem_before_cache,
        "mem_after_cache_mb": mem_after_cache,
        "mem_after_prefill_mb": mem_after_prefill,
        "mem_peak_mb": mem_peak,
        "mem_final_mb": mem_final,
        "mem_cache_delta_mb": (mem_after_cache - mem_before_cache) if mem_before_cache and mem_after_cache else None,
        "mem_total_delta_mb": (mem_final - mem_before_cache) if mem_before_cache and mem_final else None,
    }

    # KV Cache specific metrics
    if mode == "paged" and cache_mgr:
        stats = cache_mgr.get_cache_stats()
        result["paged_blocks_used"] = stats["num_used_blocks"]
        result["paged_blocks_total"] = stats["max_num_blocks"]
        result["paged_utilization_pct"] = stats["utilization"]
        result["paged_block_size"] = block_size
        # Actual memory used by KV data
        actual_tokens = stats["sequences"][0]["length"] if 0 in stats["sequences"] else 0
        num_layers = stats["num_layers"]
        kv_heads = cache_mgr.num_kv_heads
        head_dim = cache_mgr.head_dim
        result["kv_actual_tokens"] = actual_tokens
        result["kv_actual_memory_mb"] = (actual_tokens * kv_heads * head_dim * 4 * 2 * num_layers) / (1024**2)
        result["kv_reserved_memory_mb"] = (max_blocks * block_size * kv_heads * head_dim * 4 * 2 * num_layers) / (1024**2)
    else:
        # Contiguous cache
        result["kv_actual_tokens"] = prompt_len + num_generated
        num_layers_est = 28  # DeepSeek-R1-Distill-Qwen-1.5B
        kv_heads_est = 2
        head_dim_est = 96
        result["kv_reserved_memory_mb"] = (max_cache_len * kv_heads_est * head_dim_est * 4 * 2 * num_layers_est) / (1024**2)
        result["kv_actual_memory_mb"] = ((prompt_len + num_generated) * kv_heads_est * head_dim_est * 4 * 2 * num_layers_est) / (1024**2)

    return result


# ─────────────────────────────────────────────────────────────
#  Report Generation
# ─────────────────────────────────────────────────────────────

def format_report(contig_result, paged_result, config, args):
    """Generate a comprehensive Markdown report."""
    
    cr = contig_result
    pr = paged_result
    
    def ratio(a, b):
        if b and b > 0:
            return f"{a/b:.2f}x"
        return "N/A"
    
    def fmt_ms(v):
        if v is None:
            return "N/A"
        return f"{v:.2f}"
    
    def fmt_mb(v):
        if v is None:
            return "N/A"
        return f"{v:.1f}"
    
    lines = []
    lines.append("# Paged Attention vs Contiguous KV Cache — 性能对比报告")
    lines.append("")
    lines.append(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**设备**: {args.device}")
    lines.append(f"**模型**: DeepSeek-R1-Distill-Qwen-1.5B")
    lines.append("")
    
    # Model config
    lines.append("## 1. 模型配置")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| hidden_size | {config['hidden_size']} |")
    lines.append(f"| num_hidden_layers | {config['num_hidden_layers']} |")
    lines.append(f"| num_attention_heads | {config['num_attention_heads']} |")
    lines.append(f"| num_key_value_heads | {config['num_key_value_heads']} |")
    lines.append(f"| intermediate_size | {config['intermediate_size']} |")
    lines.append(f"| vocab_size | {config['vocab_size']} |")
    lines.append(f"| head_dim | {config['hidden_size'] // config['num_attention_heads']} |")
    lines.append("")
    
    # Test config
    lines.append("## 2. 测试配置")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| Prompt | `{args.prompt[:60]}{'...' if len(args.prompt) > 60 else ''}` |")
    lines.append(f"| Prompt tokens | {cr['prompt_len']} |")
    lines.append(f"| Max new tokens | {args.max_new_tokens} |")
    lines.append(f"| Sampling | Greedy (deterministic) |")
    lines.append(f"| Paged block_size | {args.block_size} |")
    lines.append(f"| Paged max_blocks | {args.max_blocks} |")
    lines.append("")
    
    # Main comparison table
    lines.append("## 3. 核心性能指标对比")
    lines.append("")
    lines.append("| 指标 | Contiguous | Paged | Paged/Contig |")
    lines.append("|------|-----------|-------|-------------|")
    
    lines.append(f"| **Prefill 延迟 (ms)** | {fmt_ms(cr['prefill_time_ms'])} | {fmt_ms(pr['prefill_time_ms'])} | {ratio(pr['prefill_time_ms'], cr['prefill_time_ms'])} |")
    lines.append(f"| **Prefill 吞吐 (tok/s)** | {cr['prefill_throughput_tok_s']:.1f} | {pr['prefill_throughput_tok_s']:.1f} | {ratio(pr['prefill_throughput_tok_s'], cr['prefill_throughput_tok_s'])} |")
    lines.append(f"| **TTFT (ms)** | {fmt_ms(cr['ttft_ms'])} | {fmt_ms(pr['ttft_ms'])} | {ratio(pr['ttft_ms'], cr['ttft_ms'])} |")
    lines.append(f"| **Decode 平均延迟 (ms/tok)** | {fmt_ms(cr['decode_avg_ms'])} | {fmt_ms(pr['decode_avg_ms'])} | {ratio(pr['decode_avg_ms'], cr['decode_avg_ms'])} |")
    lines.append(f"| **Decode 中位数延迟 (ms/tok)** | {fmt_ms(cr['decode_median_ms'])} | {fmt_ms(pr['decode_median_ms'])} | {ratio(pr['decode_median_ms'], cr['decode_median_ms'])} |")
    lines.append(f"| **Decode P90 延迟 (ms/tok)** | {fmt_ms(cr['decode_p90_ms'])} | {fmt_ms(pr['decode_p90_ms'])} | {ratio(pr['decode_p90_ms'], cr['decode_p90_ms'])} |")
    lines.append(f"| **Decode P99 延迟 (ms/tok)** | {fmt_ms(cr['decode_p99_ms'])} | {fmt_ms(pr['decode_p99_ms'])} | {ratio(pr['decode_p99_ms'], cr['decode_p99_ms'])} |")
    lines.append(f"| **Decode 吞吐 (tok/s)** | {cr['decode_throughput_tok_s']:.2f} | {pr['decode_throughput_tok_s']:.2f} | {ratio(pr['decode_throughput_tok_s'], cr['decode_throughput_tok_s'])} |")
    lines.append(f"| **整体吞吐 (tok/s)** | {cr['overall_throughput_tok_s']:.2f} | {pr['overall_throughput_tok_s']:.2f} | {ratio(pr['overall_throughput_tok_s'], cr['overall_throughput_tok_s'])} |")
    lines.append(f"| **总生成时间 (s)** | {cr['total_time_s']:.2f} | {pr['total_time_s']:.2f} | {ratio(pr['total_time_s'], cr['total_time_s'])} |")
    lines.append(f"| **生成 token 数** | {cr['num_generated']} | {pr['num_generated']} | — |")
    lines.append("")
    
    # Decode latency distribution
    lines.append("## 4. Decode 延迟分布")
    lines.append("")
    lines.append("| 统计量 | Contiguous (ms) | Paged (ms) |")
    lines.append("|--------|----------------|------------|")
    lines.append(f"| Min | {fmt_ms(cr['decode_min_ms'])} | {fmt_ms(pr['decode_min_ms'])} |")
    lines.append(f"| Mean | {fmt_ms(cr['decode_avg_ms'])} | {fmt_ms(pr['decode_avg_ms'])} |")
    lines.append(f"| Median | {fmt_ms(cr['decode_median_ms'])} | {fmt_ms(pr['decode_median_ms'])} |")
    lines.append(f"| P90 | {fmt_ms(cr['decode_p90_ms'])} | {fmt_ms(pr['decode_p90_ms'])} |")
    lines.append(f"| P99 | {fmt_ms(cr['decode_p99_ms'])} | {fmt_ms(pr['decode_p99_ms'])} |")
    lines.append(f"| Max | {fmt_ms(cr['decode_max_ms'])} | {fmt_ms(pr['decode_max_ms'])} |")
    lines.append(f"| Std Dev | {fmt_ms(cr['decode_std_ms'])} | {fmt_ms(pr['decode_std_ms'])} |")
    lines.append("")
    
    # Decode latency trend
    lines.append("## 5. Decode 延迟趋势（随序列长度变化）")
    lines.append("")
    lines.append("展示 decode 过程中不同阶段的平均延迟，用于观察随着 KV cache 增大延迟是否上升。")
    lines.append("")
    if cr['decode_segments'] and pr['decode_segments']:
        seg_keys = list(cr['decode_segments'].keys())
        lines.append("| 阶段 | Contiguous (ms) | Paged (ms) | Paged/Contig |")
        lines.append("|------|----------------|------------|-------------|")
        for key in seg_keys:
            c_val = cr['decode_segments'].get(key, 0)
            p_val = pr['decode_segments'].get(key, 0)
            lines.append(f"| {key} | {c_val:.2f} | {p_val:.2f} | {ratio(p_val, c_val)} |")
        lines.append("")
    
    # Per-step latency data (first 10, last 10)
    lines.append("### 逐步延迟采样")
    lines.append("")
    if cr['decode_times_ms'] and pr['decode_times_ms']:
        n_c = len(cr['decode_times_ms'])
        n_p = len(pr['decode_times_ms'])
        n_show = min(10, n_c, n_p)
        
        lines.append("**前 10 步:**")
        lines.append("")
        lines.append("| Step | Seq Len | Contiguous (ms) | Paged (ms) |")
        lines.append("|------|---------|----------------|------------|")
        for i in range(n_show):
            seq_len = cr['prompt_len'] + i + 1
            lines.append(f"| {i+1} | {seq_len} | {cr['decode_times_ms'][i]:.2f} | {pr['decode_times_ms'][i]:.2f} |")
        lines.append("")
        
        if n_c > 20 and n_p > 20:
            lines.append("**后 10 步:**")
            lines.append("")
            lines.append("| Step | Seq Len | Contiguous (ms) | Paged (ms) |")
            lines.append("|------|---------|----------------|------------|")
            for i in range(max(n_c - 10, 0), n_c):
                if i < n_p:
                    seq_len = cr['prompt_len'] + i + 1
                    lines.append(f"| {i+1} | {seq_len} | {cr['decode_times_ms'][i]:.2f} | {pr['decode_times_ms'][i]:.2f} |")
            lines.append("")
    
    # GPU memory
    if cr['mem_before_cache_mb'] is not None:
        lines.append("## 6. GPU 显存使用")
        lines.append("")
        lines.append("| 阶段 | Contiguous (MB) | Paged (MB) | 差值 (MB) |")
        lines.append("|------|----------------|------------|----------|")
        lines.append(f"| Cache 初始化前 | {fmt_mb(cr['mem_before_cache_mb'])} | {fmt_mb(pr['mem_before_cache_mb'])} | — |")
        lines.append(f"| Cache 初始化后 | {fmt_mb(cr['mem_after_cache_mb'])} | {fmt_mb(pr['mem_after_cache_mb'])} | {fmt_mb((pr['mem_after_cache_mb'] or 0) - (cr['mem_after_cache_mb'] or 0))} |")
        lines.append(f"| Prefill 后 | {fmt_mb(cr['mem_after_prefill_mb'])} | {fmt_mb(pr['mem_after_prefill_mb'])} | {fmt_mb((pr['mem_after_prefill_mb'] or 0) - (cr['mem_after_prefill_mb'] or 0))} |")
        lines.append(f"| 峰值 | {fmt_mb(cr['mem_peak_mb'])} | {fmt_mb(pr['mem_peak_mb'])} | {fmt_mb((pr['mem_peak_mb'] or 0) - (cr['mem_peak_mb'] or 0))} |")
        lines.append(f"| 最终 | {fmt_mb(cr['mem_final_mb'])} | {fmt_mb(pr['mem_final_mb'])} | {fmt_mb((pr['mem_final_mb'] or 0) - (cr['mem_final_mb'] or 0))} |")
        lines.append(f"| Cache 分配增量 | {fmt_mb(cr['mem_cache_delta_mb'])} | {fmt_mb(pr['mem_cache_delta_mb'])} | — |")
        lines.append(f"| 总增量 | {fmt_mb(cr['mem_total_delta_mb'])} | {fmt_mb(pr['mem_total_delta_mb'])} | — |")
        lines.append("")
    
    # KV Cache memory analysis
    lines.append("## 7. KV Cache 内存效率分析")
    lines.append("")
    lines.append("| 指标 | Contiguous | Paged |")
    lines.append("|------|-----------|-------|")
    lines.append(f"| 实际缓存 token 数 | {cr['kv_actual_tokens']} | {pr.get('kv_actual_tokens', 'N/A')} |")
    lines.append(f"| KV 实际使用内存 (MB) | {cr['kv_actual_memory_mb']:.2f} | {pr['kv_actual_memory_mb']:.2f} |")
    lines.append(f"| KV 预留内存 (MB) | {cr['kv_reserved_memory_mb']:.2f} | {pr.get('kv_reserved_memory_mb', 0):.2f} |")
    if cr['kv_reserved_memory_mb'] > 0:
        lines.append(f"| 内存利用率 | {cr['kv_actual_memory_mb']/cr['kv_reserved_memory_mb']*100:.1f}% | {pr['kv_actual_memory_mb']/max(pr.get('kv_reserved_memory_mb',1),1e-9)*100:.1f}% |")
    if 'paged_blocks_used' in pr:
        lines.append(f"| 使用 blocks / 总 blocks | — | {pr['paged_blocks_used']}/{pr['paged_blocks_total']} |")
        lines.append(f"| Block 利用率 | — | {pr['paged_utilization_pct']:.1f}% |")
        lines.append(f"| Block 大小 | — | {pr['paged_block_size']} tokens |")
    lines.append("")
    
    # Analysis / Conclusions
    lines.append("## 8. 分析与结论")
    lines.append("")
    
    # Decode speed comparison
    if cr['decode_avg_ms'] > 0 and pr['decode_avg_ms'] > 0:
        speed_ratio = pr['decode_avg_ms'] / cr['decode_avg_ms']
        if speed_ratio < 0.95:
            lines.append(f"- **Decode 速度**: Paged Attention 比 Contiguous **快 {(1-speed_ratio)*100:.1f}%**（平均每 token）")
        elif speed_ratio > 1.05:
            lines.append(f"- **Decode 速度**: Paged Attention 比 Contiguous **慢 {(speed_ratio-1)*100:.1f}%**（平均每 token）")
        else:
            lines.append(f"- **Decode 速度**: 两者基本持平（Paged/Contiguous = {speed_ratio:.2f}x）")
    
    # Latency trend
    if cr['decode_segments'] and pr['decode_segments']:
        seg_keys = list(cr['decode_segments'].keys())
        if len(seg_keys) >= 2:
            cr_first = cr['decode_segments'][seg_keys[0]]
            cr_last = cr['decode_segments'][seg_keys[-1]]
            pr_first = pr['decode_segments'][seg_keys[0]]
            pr_last = pr['decode_segments'][seg_keys[-1]]
            if cr_first > 0:
                cr_growth = (cr_last - cr_first) / cr_first * 100
                lines.append(f"- **Contiguous 延迟增长**: 从 {cr_first:.1f}ms 到 {cr_last:.1f}ms（+{cr_growth:.1f}%）")
            if pr_first > 0:
                pr_growth = (pr_last - pr_first) / pr_first * 100
                lines.append(f"- **Paged 延迟增长**: 从 {pr_first:.1f}ms 到 {pr_last:.1f}ms（+{pr_growth:.1f}%）")
    
    # Memory
    if cr['mem_cache_delta_mb'] is not None and pr['mem_cache_delta_mb'] is not None:
        lines.append(f"- **Cache 初始化显存**: Contiguous {cr['mem_cache_delta_mb']:.0f}MB vs Paged {pr['mem_cache_delta_mb']:.0f}MB")
    
    lines.append(f"- **KV Cache 预留 vs 实际**: Contiguous 预留 {cr['kv_reserved_memory_mb']:.1f}MB, "
                 f"Paged 预留 {pr.get('kv_reserved_memory_mb', 0):.1f}MB")
    lines.append("")
    
    # Key insights
    lines.append("### 关键洞察")
    lines.append("")
    lines.append("1. **Contiguous Cache**: 在 `init_cache()` 时一次性预分配 `max_cache_len` 大小的连续内存。"
                 "优点是 cache 读写是简单的 slice 操作（零额外开销）；缺点是无论实际生成多少 token，内存都已占满。")
    lines.append("")
    lines.append("2. **Paged Cache**: 在 `__init__` 时预分配 `max_blocks` 个物理块池。"
                 "虽然物理块也是预分配的，但 page table 是按需增长的。"
                 "每步 decode 需要额外的 gather 操作（从分散的 block 中收集成连续张量），"
                 "这带来了额外的计算开销。")
    lines.append("")
    lines.append("3. **延迟随序列增长**: 两种模式下 decode 延迟都会随序列变长而增加，"
                 "这是因为 attention 计算复杂度为 O(seq_len)。"
                 "Paged 模式的 gather 开销也会随 seq_len 线性增长。")
    lines.append("")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Paged vs Contiguous KV Cache Benchmark")
    parser.add_argument("--model_path", default="/home/iclab/LLM_ndl/llaisys/models/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--prompt", default="Please explain what is PagedAttention in vLLM and how it improves memory efficiency for large language model serving.")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "cpu_numpy"])
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--max_blocks", type=int, default=256)
    parser.add_argument("--greedy", action="store_true", default=True, help="Always greedy (deterministic)")
    parser.add_argument("--output", default="doc/bench_paged_vs_contiguous_report.md",
                        help="Output report path")
    args = parser.parse_args()

    is_cuda = (args.device == "cuda")

    print("=" * 70)
    print("  Paged Attention vs Contiguous KV Cache — Performance Benchmark")
    print("=" * 70)
    print()

    # ── Load model (once) ──
    print(f"[1/5] Loading model on {args.device}...")
    device = get_device(args.device)
    t0 = time.time()
    model, tokenizer, config = load_model_and_tokenizer(args.model_path, device)
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    print(f"  Config: layers={config['num_hidden_layers']}, "
          f"heads={config['num_attention_heads']}, "
          f"kv_heads={config['num_key_value_heads']}, "
          f"hidden={config['hidden_size']}")

    # ── Tokenize ──
    messages = [{"role": "user", "content": args.prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(chat_text)
    eos_token_id = config.get("eos_token_id", 151643)
    print(f"  Prompt: \"{args.prompt[:60]}...\"")
    print(f"  Prompt tokens: {len(input_ids)}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print()

    # ── Warmup (optional short run to stabilize GPU clocks) ──
    if is_cuda:
        print("[2/5] GPU warmup (short prefill)...")
        model.reset_cache()
        model.eval()
        model.init_cache(batch_size=1, max_cache_len=len(input_ids) + 5)
        with no_grad():
            ids_t = Tensor(np.array([input_ids], dtype=np.float32),
                          device=device, dtype="float32", requires_grad=False)
            _ = model(ids_t, start_pos=0, last_only=True)
        sync_cuda_device()
        model.reset_cache()
        print("  Warmup done.")
        print()

    # ── Run Contiguous benchmark ──
    print(f"[3/5] Running CONTIGUOUS benchmark ({args.max_new_tokens} tokens)...")
    if is_cuda:
        sync_cuda_device()
    contig_result = run_benchmark(
        model, input_ids, args.max_new_tokens, eos_token_id,
        use_paged=False, is_cuda=is_cuda
    )
    print(f"  Done: {contig_result['num_generated']} tokens in {contig_result['total_time_s']:.2f}s "
          f"({contig_result['decode_throughput_tok_s']:.2f} tok/s decode)")
    
    # Clean up between runs
    model.reset_cache()
    if is_cuda:
        sync_cuda_device()
        time.sleep(1)  # Let GPU memory settle

    # ── Run Paged benchmark ──
    print(f"[4/5] Running PAGED benchmark ({args.max_new_tokens} tokens, "
          f"block_size={args.block_size}, max_blocks={args.max_blocks})...")
    if is_cuda:
        sync_cuda_device()
    paged_result = run_benchmark(
        model, input_ids, args.max_new_tokens, eos_token_id,
        use_paged=True, block_size=args.block_size, max_blocks=args.max_blocks,
        is_cuda=is_cuda
    )
    print(f"  Done: {paged_result['num_generated']} tokens in {paged_result['total_time_s']:.2f}s "
          f"({paged_result['decode_throughput_tok_s']:.2f} tok/s decode)")
    print()

    # ── Generate report ──
    print("[5/5] Generating report...")
    report = format_report(contig_result, paged_result, config, args)

    # Ensure output directory exists
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', args.output)
    output_path = os.path.normpath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report saved to: {output_path}")
    print()

    # ── Print summary to console ──
    print("=" * 70)
    print("  QUICK SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<35s} {'Contiguous':>12s} {'Paged':>12s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    print(f"  {'Prefill (ms)':<35s} {contig_result['prefill_time_ms']:>12.1f} {paged_result['prefill_time_ms']:>12.1f}")
    print(f"  {'TTFT (ms)':<35s} {contig_result['ttft_ms']:>12.1f} {paged_result['ttft_ms']:>12.1f}")
    print(f"  {'Decode avg (ms/tok)':<35s} {contig_result['decode_avg_ms']:>12.2f} {paged_result['decode_avg_ms']:>12.2f}")
    print(f"  {'Decode throughput (tok/s)':<35s} {contig_result['decode_throughput_tok_s']:>12.2f} {paged_result['decode_throughput_tok_s']:>12.2f}")
    print(f"  {'Total time (s)':<35s} {contig_result['total_time_s']:>12.2f} {paged_result['total_time_s']:>12.2f}")
    print(f"  {'Tokens generated':<35s} {contig_result['num_generated']:>12d} {paged_result['num_generated']:>12d}")
    if contig_result['mem_peak_mb']:
        print(f"  {'GPU peak memory (MB)':<35s} {contig_result['mem_peak_mb']:>12.0f} {paged_result['mem_peak_mb']:>12.0f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
