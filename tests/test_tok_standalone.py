"""Standalone tokenizer test - no uniti framework import needed.
Usage: python tests/test_tok_standalone.py /path/to/model
"""
import sys, os, json, re

# Only import the tokenizer module directly (no uniti/__init__.py)
tok_module_path = os.path.join(os.path.dirname(__file__), '..', 'python', 'uniti')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Monkey-patch: prevent uniti/__init__.py from loading C extensions
# We only need the tokenizer module
import importlib
import types
# Create a fake uniti module to avoid __init__.py side effects
fake_uniti = types.ModuleType('uniti')
fake_uniti.__path__ = [tok_module_path]
sys.modules['uniti'] = fake_uniti

from uniti.tokenizer import UniTITokenizer, _HAS_REGEX, _compile_pattern, _convert_unicode_props

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/mnt/cephfs/szcsp1/lubaninfra/all_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

print(f"Model path: {MODEL}")
print(f"Has regex module: {_HAS_REGEX}")

# ─── 1. Check tokenizer.json structure ───
print("\n" + "=" * 60)
print("1. tokenizer.json structure")
print("=" * 60)
tok_path = os.path.join(MODEL, "tokenizer.json")
with open(tok_path, "r") as f:
    data = json.load(f)

pre_tok = data.get("pre_tokenizer", {})
print("pre_tokenizer type:", pre_tok.get("type"))
if pre_tok.get("type") == "Sequence":
    for item in pre_tok.get("pretokenizers", []):
        print(f"  - {item.get('type')}", end="")
        if "pattern" in item:
            pat = item["pattern"]
            if isinstance(pat, dict) and "Regex" in pat:
                print(f"  pattern={pat['Regex'][:80]!r}", end="")
        print()

decoder_info = data.get("decoder", {})
print(f"decoder type: {decoder_info.get('type')}")
if decoder_info.get("type") == "Sequence":
    for d in decoder_info.get("decoders", []):
        print(f"  - {d.get('type')}")

vocab = data.get("model", {}).get("vocab", {})
byte_marker_count = sum(1 for k in vocab if "\u0120" in k)  # Ġ = chr(288) = \u0120
print(f"\nvocab size: {len(vocab)}")
print(f"Vocab keys with Ġ (byte-level marker): {byte_marker_count}")

# Show some vocab samples
print("\nSample vocab (first 15):")
for i, (k, v) in enumerate(list(vocab.items())[:15]):
    print(f"  {v:>6}: {k!r}")

# Check specific tokens
for tok_str in ["Hello", "hello", "ĠHello", "Ġhello", " Hello", " hello", ",", "Ġ,", "Ġhow"]:
    if tok_str in vocab:
        print(f"  ** vocab[{tok_str!r}] = {vocab[tok_str]}")

# ─── 2. Load our tokenizer ───
print("\n" + "=" * 60)
print("2. UniTITokenizer")
print("=" * 60)
tok = UniTITokenizer.from_pretrained(MODEL)
print(f"Loaded: {tok}")
print(f"byte_level: {tok._byte_level}")
print(f"eos_token={tok.eos_token!r}  eos_token_id={tok.eos_token_id}")
print(f"Pattern module: {type(tok._pattern).__module__}")
print(f"Pattern: {tok._pattern.pattern[:200]!r}")

# Test regex matching
test_texts = [
    "Hello, how are you?",
    "user\nHello, how are you?",
    "\n",
]
for t in test_texts:
    matches = tok._pattern.findall(t)
    print(f"  findall({t!r}): {matches}")

# ─── 3. Encode tests ───
print("\n" + "=" * 60)
print("3. Encode tests")
print("=" * 60)

test = "Hello, how are you?"
our_ids = tok.encode(test)
print(f"\nEncode '{test}':")
print(f"  IDs ({len(our_ids)}): {our_ids}")
print(f"  Decoded: {tok.decode(our_ids, skip_special_tokens=False)!r}")
print(f"  Token breakdown:")
for i, tid in enumerate(our_ids):
    tok_str = tok.id_to_token.get(tid, '???')
    print(f"    [{i:2d}] id={tid:>6}  token={tok_str!r}")

# Chat text
msgs = [{"role": "user", "content": test}]
chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(f"\nChat text: {chat_text!r}")

chat_ids = tok.encode(chat_text)
print(f"Chat IDs ({len(chat_ids)}): {chat_ids}")
print(f"  Token breakdown:")
for i, tid in enumerate(chat_ids):
    tok_str = tok.id_to_token.get(tid, '???')
    print(f"    [{i:2d}] id={tid:>6}  token={tok_str!r}")

# ─── 4. Compare with transformers ───
print("\n" + "=" * 60)
print("4. transformers reference")
print("=" * 60)
try:
    from transformers import AutoTokenizer
    ref = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    ref_simple = ref.encode(test, add_special_tokens=False)
    print(f"Simple encode ({len(ref_simple)} tokens): {ref_simple}")

    ref_chat = ref.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    ref_chat_text = ref.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"Chat text: {ref_chat_text!r}")
    print(f"Chat IDs ({len(ref_chat)} tokens): {list(ref_chat)}")

    print(f"\nRef simple token breakdown:")
    for i, tid in enumerate(ref_simple):
        tok_str = ref.convert_ids_to_tokens([tid])[0]
        print(f"    [{i:2d}] id={tid:>6}  token={tok_str!r}")

    print("\n--- Comparison ---")
    print(f"Simple match: {our_ids == ref_simple}")
    print(f"Chat match:   {chat_ids == list(ref_chat)}")

    if our_ids != ref_simple:
        print("\n  MISMATCH in simple encode:")
        max_len = max(len(our_ids), len(ref_simple))
        for i in range(min(max_len, 30)):
            a = our_ids[i] if i < len(our_ids) else "---"
            b = ref_simple[i] if i < len(ref_simple) else "---"
            marker = " <<<" if a != b else ""
            a_tok = tok.id_to_token.get(a, '?') if isinstance(a, int) else '?'
            b_tok = ref.convert_ids_to_tokens([b])[0] if isinstance(b, int) else '?'
            print(f"    [{i:2d}] ours={str(a):>6} ({a_tok!r:>20})  ref={str(b):>6} ({b_tok!r:>20}){marker}")

    if chat_ids != list(ref_chat):
        print("\n  MISMATCH in chat encode:")
        max_len = max(len(chat_ids), len(ref_chat))
        for i in range(min(max_len, 40)):
            a = chat_ids[i] if i < len(chat_ids) else "---"
            b = ref_chat[i] if i < len(ref_chat) else "---"
            marker = " <<<" if a != b else ""
            a_tok = tok.id_to_token.get(a, '?') if isinstance(a, int) else '?'
            b_tok = ref.convert_ids_to_tokens([b])[0] if isinstance(b, int) else '?'
            print(f"    [{i:2d}] ours={str(a):>6} ({a_tok!r:>20})  ref={str(b):>6} ({b_tok!r:>20}){marker}")
        print(f"  Length: ours={len(chat_ids)} ref={len(ref_chat)}")

except ImportError:
    print("transformers not available, skipping comparison")
except Exception as e:
    print(f"Error: {e}")
    import traceback; traceback.print_exc()
