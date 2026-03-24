"""Debug script: compare UniTITokenizer vs transformers tokenizer."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/mnt/cephfs/lubaninfra/all_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# ─── 1. Inspect tokenizer.json structure ───
print("=" * 60)
print("1. tokenizer.json structure")
print("=" * 60)
tok_path = os.path.join(MODEL, "tokenizer.json")
if not os.path.exists(tok_path):
    print(f"tokenizer.json not found at {tok_path}")
    sys.exit(1)

with open(tok_path, "r") as f:
    data = json.load(f)

pre_tok = data.get("pre_tokenizer", {})
print("pre_tokenizer:", json.dumps(pre_tok, indent=2, ensure_ascii=False)[:2000])
print("\nnormalizer:", json.dumps(data.get("normalizer"), indent=2))
print("\ndecoder:", json.dumps(data.get("decoder", {}), indent=2)[:500])
print("\nmodel type:", data.get("model", {}).get("type"))

vocab = data.get("model", {}).get("vocab", {})
print(f"\nvocab size: {len(vocab)}")

# Check byte-level heuristic: count "Ġ" in vocab keys
byte_marker_count = sum(1 for k in vocab if "Ġ" in k)
print(f"Vocab keys containing 'Ġ' (chr(288)): {byte_marker_count}")

# Sample some vocab entries
print("\nSample vocab entries:")
for i, (k, v) in enumerate(vocab.items()):
    if i < 10:
        print(f"  {v}: {k!r}")
    elif i == 10:
        print("  ...")
    if k in ("Hello", "hello", " Hello", " hello", "Ġhello", "ĠHello"):
        print(f"  ** FOUND: {v}: {k!r}")

# Check special tokens
for tok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
    if tok in vocab:
        print(f"  vocab[{tok!r}] = {vocab[tok]}")

added = data.get("added_tokens", [])
print(f"\nadded_tokens count: {len(added)}")
for t in added[:10]:
    print(f"  id={t['id']}  content={t['content']!r}  special={t.get('special')}")

# ─── 2. Our tokenizer encode ───
print("\n" + "=" * 60)
print("2. UniTITokenizer")
print("=" * 60)
from uniti.tokenizer import UniTITokenizer
tok = UniTITokenizer.from_pretrained(MODEL)
print(f"Loaded: {tok}")
print(f"byte_level: {tok._byte_level}")
print(f"eos_token={tok.eos_token!r}  eos_token_id={tok.eos_token_id}")

# Simple text encode
test = "Hello, how are you?"
our_ids = tok.encode(test)
print(f"\nEncode '{test}' ({len(our_ids)} tokens): {our_ids}")
print(f"  Decoded back: {tok.decode(our_ids, skip_special_tokens=False)!r}")

# Chat template
msgs = [{"role": "user", "content": test}]
chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(f"\nChat text: {chat_text!r}")
chat_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
print(f"Chat IDs ({len(chat_ids)} tokens): {chat_ids}")

# ─── 3. Compare with transformers ───
print("\n" + "=" * 60)
print("3. transformers reference")
print("=" * 60)
try:
    from transformers import AutoTokenizer
    ref = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    ref_simple = ref.encode(test, add_special_tokens=False)
    print(f"Encode '{test}' ({len(ref_simple)} tokens): {ref_simple}")

    ref_chat = ref.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    print(f"Chat IDs ({len(ref_chat)} tokens): {ref_chat}")

    # Show chat text from ref
    ref_chat_text = ref.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"Chat text: {ref_chat_text!r}")

    print("\n--- Comparison ---")
    print(f"Simple encode match: {our_ids == ref_simple}")
    print(f"Chat encode match:   {chat_ids == ref_chat}")

    if our_ids != ref_simple:
        print("\nSimple encode MISMATCH:")
        for i in range(max(len(our_ids), len(ref_simple))):
            a = our_ids[i] if i < len(our_ids) else "---"
            b = ref_simple[i] if i < len(ref_simple) else "---"
            if a != b:
                a_tok = tok.decode([a]) if isinstance(a, int) else "?"
                b_tok = ref.decode([b]) if isinstance(b, int) else "?"
                print(f"  [{i}] ours={a} ({a_tok!r})  ref={b} ({b_tok!r})")

    if chat_ids != ref_chat:
        print("\nChat encode MISMATCH:")
        for i in range(min(max(len(chat_ids), len(ref_chat)), 50)):
            a = chat_ids[i] if i < len(chat_ids) else "---"
            b = ref_chat[i] if i < len(ref_chat) else "---"
            marker = " ***" if a != b else ""
            a_tok = tok.decode([a]) if isinstance(a, int) else "?"
            b_tok = ref.decode([b]) if isinstance(b, int) else "?"
            print(f"  [{i:2d}] ours={a:>6} ({a_tok!r:>15})  ref={b:>6} ({b_tok!r:>15}){marker}")
        if len(chat_ids) != len(ref_chat):
            print(f"  Length: ours={len(chat_ids)} ref={len(ref_chat)}")
except ImportError:
    print("transformers not available, skipping comparison")
