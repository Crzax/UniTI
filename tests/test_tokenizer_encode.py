"""Debug script: compare UniTITokenizer vs transformers tokenizer.
Run on the machine with model access:
  python tests/test_tokenizer_encode.py /path/to/model
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/mnt/cephfs/lubaninfra/all_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# ─── 1. Our tokenizer ───
print("=" * 60)
print("1. UniTITokenizer")
print("=" * 60)
from uniti.tokenizer import UniTITokenizer, _HAS_REGEX, _compile_pattern, _convert_unicode_props
print(f"Has regex module: {_HAS_REGEX}")

tok = UniTITokenizer.from_pretrained(MODEL)
print(f"Loaded: {tok}")
print(f"byte_level: {tok._byte_level}")
print(f"eos_token={tok.eos_token!r}  eos_token_id={tok.eos_token_id}")
print(f"Pattern type: {type(tok._pattern).__module__}.{type(tok._pattern).__name__}")
print(f"Pattern: {tok._pattern.pattern[:200]!r}")

# Test regex matching
import re
test_texts = [
    "Hello, how are you?",
    "user\nHello",
    "\n",
    "assistant\n",
]
for t in test_texts:
    matches = tok._pattern.findall(t)
    print(f"\n  findall({t!r}): {matches}")

# ─── 2. Encode tests ───
print("\n" + "=" * 60)
print("2. Encode tests")
print("=" * 60)

# Simple text
test = "Hello, how are you?"
our_ids = tok.encode(test)
print(f"\nEncode '{test}':")
print(f"  IDs ({len(our_ids)}): {our_ids}")
print(f"  Decoded: {tok.decode(our_ids, skip_special_tokens=False)!r}")

# Show token-by-token
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

# ─── 3. Compare with transformers ───
print("\n" + "=" * 60)
print("3. transformers reference")
print("=" * 60)
try:
    from transformers import AutoTokenizer
    ref = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    ref_simple = ref.encode(test, add_special_tokens=False)
    print(f"Simple encode ({len(ref_simple)} tokens): {ref_simple}")
    
    ref_chat = ref.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    ref_chat_text = ref.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"Chat text: {ref_chat_text!r}")
    print(f"Chat IDs ({len(ref_chat)} tokens): {ref_chat}")

    # Show ref token breakdown
    print(f"\n  Ref token breakdown for simple encode:")
    for i, tid in enumerate(ref_simple):
        tok_str = ref.convert_ids_to_tokens([tid])[0]
        print(f"    [{i:2d}] id={tid:>6}  token={tok_str!r}")

    print("\n--- Comparison ---")
    print(f"Simple match: {our_ids == ref_simple}")
    print(f"Chat match:   {chat_ids == list(ref_chat)}")

    if our_ids != ref_simple:
        print("\n  MISMATCH in simple encode:")
        max_len = max(len(our_ids), len(ref_simple))
        for i in range(min(max_len, 50)):
            a = our_ids[i] if i < len(our_ids) else "---"
            b = ref_simple[i] if i < len(ref_simple) else "---"
            marker = " <<<" if a != b else ""
            a_tok = tok.id_to_token.get(a, '?') if isinstance(a, int) else '?'
            b_tok = ref.convert_ids_to_tokens([b])[0] if isinstance(b, int) else '?'
            print(f"    [{i:2d}] ours={str(a):>6} ({a_tok!r:>20})  ref={str(b):>6} ({b_tok!r:>20}){marker}")

    if chat_ids != list(ref_chat):
        print("\n  MISMATCH in chat encode:")
        max_len = max(len(chat_ids), len(ref_chat))
        for i in range(min(max_len, 50)):
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
