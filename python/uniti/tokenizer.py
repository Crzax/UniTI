"""
Lightweight BPE tokenizer for Qwen2 / DeepSeek-R1-Distill models.

Loads the HuggingFace ``tokenizer.json`` directly and implements:
  - BPE encode  (text → token IDs)
  - decode      (token IDs → text)
  - chat template application

Zero external dependencies beyond the Python standard library.
"""

import json
import re
import os
from typing import Dict, List, Optional, Tuple, Union

# Prefer the `regex` module for Unicode \p{} support; fall back to stdlib `re`.
try:
    import regex as _re_engine
    _HAS_REGEX = True
except ImportError:
    _re_engine = re  # type: ignore[assignment]
    _HAS_REGEX = False


class UniTITokenizer:
    """A self-contained BPE tokenizer that reads HuggingFace tokenizer.json.

    Supports Qwen2 / DeepSeek / GPT-style byte-level BPE tokenizers.

    Usage
    -----
    >>> tok = UniTITokenizer.from_pretrained("/path/to/model")
    >>> ids = tok.encode("Hello, world!")
    >>> text = tok.decode(ids)
    >>> # Chat template
    >>> ids = tok.apply_chat_template([{"role": "user", "content": "Hi"}])
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        added_tokens: Optional[Dict[str, int]] = None,
        special_tokens: Optional[Dict[str, int]] = None,
        pattern: Optional[str] = None,
        chat_template: Optional[str] = None,
        eos_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        byte_level: bool = True,
    ):
        self.vocab = vocab
        self.id_to_token: Dict[int, str] = {v: k for k, v in vocab.items()}

        # Added tokens (like <|im_start|>) with their IDs
        self.added_tokens = added_tokens or {}
        self.added_token_ids: Dict[int, str] = {v: k for k, v in self.added_tokens.items()}

        # Merge all into id_to_token for decoding
        for tok_str, tok_id in self.added_tokens.items():
            self.id_to_token[tok_id] = tok_str

        # Special tokens
        self.special_tokens = special_tokens or {}
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.eos_token_id = self.added_tokens.get(eos_token, vocab.get(eos_token)) if eos_token else None
        self.chat_template = chat_template

        # BPE merges: list of (a, b) pairs, priority = index
        self.merges = merges
        self.bpe_ranks: Dict[Tuple[str, str], int] = {pair: i for i, pair in enumerate(merges)}

        # Pre-tokenization regex (GPT-2 / Qwen2 style)
        if pattern:
            self._pattern = _compile_pattern(pattern)
        else:
            default_pattern = (
                r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
                r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
            )
            self._pattern = _compile_pattern(default_pattern)

        # Detect whether vocab uses byte-level encoding (GPT-2 style)
        # or direct UTF-8 tokens (Qwen2 style).
        # Heuristic: if "Ġ" (the byte-level representation of space, chr(288))
        # appears frequently in vocab keys, it's byte-level.
        # Qwen2 vocab uses raw UTF-8 characters instead.
        self._byte_level = byte_level
        if self._byte_level:
            self._byte_encoder = self._build_byte_encoder()
            self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
        else:
            self._byte_encoder = None
            self._byte_decoder = None

    # ─── Factory ────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, model_path: str, trust_remote_code: bool = True) -> "UniTITokenizer":
        """Load tokenizer from a HuggingFace model directory.

        Reads ``tokenizer.json`` and ``tokenizer_config.json``.
        """
        tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
        config_path = os.path.join(model_path, "tokenizer_config.json")

        if not os.path.isfile(tokenizer_json_path):
            raise FileNotFoundError(f"tokenizer.json not found in {model_path}")

        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)

        # ── Extract vocab ──
        model_data = tok_data.get("model", {})
        vocab: Dict[str, int] = model_data.get("vocab", {})

        # ── Extract merges ──
        merges_raw: List[str] = model_data.get("merges", [])
        merges = []
        for m in merges_raw:
            parts = m.split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))

        # ── Added tokens ──
        added_tokens_list = tok_data.get("added_tokens", [])
        added_tokens: Dict[str, int] = {}
        special_tokens: Dict[str, int] = {}
        for tok_info in added_tokens_list:
            content = tok_info["content"]
            tid = tok_info["id"]
            added_tokens[content] = tid
            if tok_info.get("special", False):
                special_tokens[content] = tid

        # ── Pre-tokenizer pattern ──
        pattern = None
        byte_level = False
        pre_tok = tok_data.get("pre_tokenizer", {})
        if pre_tok.get("type") == "Sequence":
            for item in pre_tok.get("pretokenizers", []):
                if item.get("type") == "Split" and "pattern" in item:
                    pat = item["pattern"]
                    if isinstance(pat, dict) and "Regex" in pat:
                        pattern = pat["Regex"]
                        # Don't break — keep scanning for ByteLevel
                elif item.get("type") == "ByteLevel":
                    byte_level = True
        elif pre_tok.get("type") == "Split" and "pattern" in pre_tok:
            pat = pre_tok["pattern"]
            if isinstance(pat, dict) and "Regex" in pat:
                pattern = pat["Regex"]
        elif pre_tok.get("type") == "ByteLevel":
            byte_level = True

        # Also check the decoder type — if it's ByteLevel, tokens are byte-encoded
        decoder_info = tok_data.get("decoder", {})
        if decoder_info.get("type") == "ByteLevel":
            byte_level = True
        elif decoder_info.get("type") == "Sequence":
            for dec_item in decoder_info.get("decoders", []):
                if dec_item.get("type") == "ByteLevel":
                    byte_level = True
                    break

        # Heuristic fallback: check if the GPT-2 byte-level marker "Ġ" (chr(288))
        # is common in vocab (it represents space in byte-level BPE).
        if not byte_level:
            byte_marker_count = sum(1 for k in vocab if "Ġ" in k)
            byte_level = byte_marker_count > 100

        # ── Chat template & special tokens from config ──
        chat_template = None
        eos_token = None
        bos_token = None
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            chat_template = config.get("chat_template")
            
            eos_raw = config.get("eos_token")
            if isinstance(eos_raw, dict):
                eos_token = eos_raw.get("content")
            elif isinstance(eos_raw, str):
                eos_token = eos_raw

            bos_raw = config.get("bos_token")
            if isinstance(bos_raw, dict):
                bos_token = bos_raw.get("content")
            elif isinstance(bos_raw, str):
                bos_token = bos_raw

        return cls(
            vocab=vocab,
            merges=merges,
            added_tokens=added_tokens,
            special_tokens=special_tokens,
            pattern=pattern,
            chat_template=chat_template,
            eos_token=eos_token,
            bos_token=bos_token,
            byte_level=byte_level,
        )

    # ─── Encode ─────────────────────────────────────────────────────────

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs using byte-level BPE.

        Parameters
        ----------
        text : str
            Input text.
        add_special_tokens : bool
            If True, prepend BOS token (if defined).

        Returns
        -------
        list[int]
            Token IDs.
        """
        ids: List[int] = []
        if add_special_tokens and self.bos_token and self.bos_token in self.added_tokens:
            ids.append(self.added_tokens[self.bos_token])

        # Check for added/special tokens in the text first
        # Build a pattern to split on added tokens
        if self.added_tokens:
            # Sort by length (longest first) to avoid partial matches
            sorted_added = sorted(self.added_tokens.keys(), key=len, reverse=True)
            escaped = [re.escape(t) for t in sorted_added]
            split_pattern = re.compile("(" + "|".join(escaped) + ")")
            parts = split_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if part in self.added_tokens:
                ids.append(self.added_tokens[part])
            else:
                ids.extend(self._encode_ordinary(part))

        return ids

    def _encode_ordinary(self, text: str) -> List[int]:
        """Encode ordinary text (no special tokens) via BPE."""
        import unicodedata
        text = unicodedata.normalize("NFC", text)
        ids: List[int] = []
        # Pre-tokenize
        for match in self._pattern.finditer(text):
            chunk = match.group(0)
            if self._byte_level:
                # GPT-2 style: convert to byte-level tokens first
                byte_tokens = tuple(self._byte_encoder[b] for b in chunk.encode("utf-8"))
            else:
                # Qwen2 style: use raw characters directly
                byte_tokens = tuple(chunk)
            # Apply BPE merges
            merged = self._bpe(byte_tokens)
            for token_str in merged:
                if token_str in self.vocab:
                    ids.append(self.vocab[token_str])
                else:
                    # Fallback: encode unknown token character by character
                    for ch in token_str:
                        if ch in self.vocab:
                            ids.append(self.vocab[ch])
                        elif self._byte_level:
                            # Try byte-level fallback
                            for b in ch.encode("utf-8"):
                                bch = self._byte_encoder.get(b)
                                if bch and bch in self.vocab:
                                    ids.append(self.vocab[bch])
        return ids

    def _bpe(self, tokens: Tuple[str, ...]) -> List[str]:
        """Apply BPE merges to a sequence of tokens."""
        if len(tokens) <= 1:
            return list(tokens)

        word = list(tokens)
        while True:
            # Find the pair with the lowest merge rank
            best_pair = None
            best_rank = float("inf")
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Merge all occurrences of the best pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

            if len(word) == 1:
                break

        return word

    # ─── Decode ─────────────────────────────────────────────────────────

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Parameters
        ----------
        ids : list[int]
            Token IDs.
        skip_special_tokens : bool
            If True, omit special tokens from the output.

        Returns
        -------
        str
        """
        if self._byte_level:
            return self._decode_byte_level(ids, skip_special_tokens)
        else:
            return self._decode_direct(ids, skip_special_tokens)

    def _decode_byte_level(self, ids: List[int], skip_special_tokens: bool) -> str:
        """Decode byte-level BPE tokens (GPT-2 style)."""
        byte_list = []
        for token_id in ids:
            token_str = self.id_to_token.get(token_id, "")
            if skip_special_tokens and token_id in self.added_token_ids:
                if self.added_token_ids[token_id] in self.special_tokens:
                    continue
            for ch in token_str:
                if ch in self._byte_decoder:
                    byte_list.append(self._byte_decoder[ch])
                else:
                    byte_list.extend(ch.encode("utf-8", errors="replace"))
        return bytes(byte_list).decode("utf-8", errors="replace")

    def token_to_bytes(self, token_id: int, skip_special_tokens: bool = True) -> bytes:
        """Convert a single token ID to its raw bytes (without UTF-8 decoding).

        This is useful for streaming output where multi-byte characters (like emoji)
        may be split across multiple tokens. Feed the returned bytes into a
        ``codecs.getincrementaldecoder('utf-8')`` to get properly decoded text.

        Parameters
        ----------
        token_id : int
            A single token ID.
        skip_special_tokens : bool
            If True, return empty bytes for special tokens.

        Returns
        -------
        bytes
        """
        if self._byte_level:
            token_str = self.id_to_token.get(token_id, "")
            if skip_special_tokens and token_id in self.added_token_ids:
                if self.added_token_ids[token_id] in self.special_tokens:
                    return b""
            byte_list = []
            for ch in token_str:
                if ch in self._byte_decoder:
                    byte_list.append(self._byte_decoder[ch])
                else:
                    byte_list.extend(ch.encode("utf-8", errors="replace"))
            return bytes(byte_list)
        else:
            # Non-byte-level: token is already valid UTF-8 text
            token_str = self.id_to_token.get(token_id, "")
            if skip_special_tokens and token_id in self.added_token_ids:
                if self.added_token_ids[token_id] in self.special_tokens:
                    return b""
            return token_str.encode("utf-8")

    def _decode_direct(self, ids: List[int], skip_special_tokens: bool) -> str:
        """Decode direct UTF-8 tokens (Qwen2 style)."""
        parts = []
        for token_id in ids:
            token_str = self.id_to_token.get(token_id, "")
            if skip_special_tokens and token_id in self.added_token_ids:
                if self.added_token_ids[token_id] in self.special_tokens:
                    continue
            parts.append(token_str)
        return "".join(parts)

    # ─── Chat Template ──────────────────────────────────────────────────

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> Union[List[int], str]:
        """Apply chat template to messages.

        Supports Qwen2 / DeepSeek / ChatML style templates.

        Parameters
        ----------
        messages : list[dict]
            List of {"role": ..., "content": ...} dicts.
        tokenize : bool
            If True, return token IDs; if False, return formatted string.
        add_generation_prompt : bool
            If True, append the assistant prompt prefix.

        Returns
        -------
        list[int] or str
        """
        text = self._render_chat_template(messages, add_generation_prompt)
        if tokenize:
            return self.encode(text)
        return text

    def _render_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Render chat template.

        Tries Jinja2 first (accurate for all models), then falls back to
        simple string-based renderers for common patterns.
        """
        if self.chat_template:
            # Try Jinja2 rendering first (handles DeepSeek-R1, Qwen2, etc.)
            rendered = self._render_jinja2(messages, add_generation_prompt)
            if rendered is not None:
                return rendered
            # Fallback: simple pattern-based rendering
            return self._render_jinja_simple(messages, add_generation_prompt)

        # No template: default ChatML format
        return self._render_chatml(messages, add_generation_prompt)

    def _render_jinja2(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> Optional[str]:
        """Render chat template using Jinja2 engine.

        Returns None if jinja2 is not available.
        """
        try:
            from jinja2 import BaseLoader, Environment, StrictUndefined
        except ImportError:
            return None

        try:
            if not self.chat_template:
                return None
            env = Environment(
                loader=BaseLoader(),
                undefined=StrictUndefined,
                keep_trailing_newline=True,
                lstrip_blocks=True,
                trim_blocks=True,
            )
            template = env.from_string(self.chat_template)
            return template.render(
                messages=messages,
                add_generation_prompt=add_generation_prompt,
                bos_token=self.bos_token or "",
                eos_token=self.eos_token or "",
            )
        except Exception:
            return None

    def _render_jinja_simple(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Simple pattern-based template renderer (no jinja2 needed).

        Handles common patterns: ChatML, Llama, etc.
        Falls back to ChatML if template is too complex.
        """
        template = self.chat_template or ""

        if "<|im_start|>" in template or "im_start" in template:
            return self._render_chatml(messages, add_generation_prompt)

        if "<|begin_of_text|>" in template:
            return self._render_llama_style(messages, add_generation_prompt)

        # Ultimate fallback: ChatML
        return self._render_chatml(messages, add_generation_prompt)

    def _render_chatml(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool,
    ) -> str:
        """Render ChatML format (Qwen2, etc.)."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _render_llama_style(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool,
    ) -> str:
        """Render Llama/Llama-2 style template."""
        parts = ["<|begin_of_text|>"]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        if add_generation_prompt:
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)

    # ─── Byte-level BPE utilities ───────────────────────────────────────

    @staticmethod
    def _build_byte_encoder() -> Dict[int, str]:
        """Build GPT-2 style byte-to-unicode mapping.

        Maps bytes 0-255 to unicode characters, ensuring all characters
        are printable and avoiding whitespace/control chars that would
        interfere with BPE.
        """
        # Printable bytes that map to themselves
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = list(bs)
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}

    # ─── Utility ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.vocab) + len(self.added_tokens)

    def __repr__(self) -> str:
        return (
            f"UniTITokenizer(vocab_size={len(self.vocab)}, "
            f"added_tokens={len(self.added_tokens)}, "
            f"merges={len(self.merges)}, "
            f"byte_level={self._byte_level})"
        )


# ─── Regex compilation helpers ─────────────────────────────────────────

# Unicode property ranges (without outer brackets) for embedding inside [...]
_UNICODE_RANGES = {
    "L": (r"a-zA-Z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\u4E00-\u9FFF"
          r"\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF\u0600-\u06FF\u0980-\u09FF"),
    "N": (r"0-9\u0660-\u0669\u06F0-\u06F9\u0966-\u096F\u09E6-\u09EF"
          r"\u00B2\u00B3\u00B9\u00BC-\u00BE\u2070-\u2079\u2080-\u2089"),
    "Z": r"\s",
    "P": r"!-/:-@\[-`{-~\u00A1-\u00BF\u2000-\u206F\u3000-\u303F",
    "S": r"\$\+<->\^`\|~\u00A2-\u00A9\u00AE-\u00B1\u00B4\u00D7\u00F7"
         r"\u2190-\u21FF\u2200-\u22FF",
}


def _convert_unicode_props(pattern: str) -> str:
    r"""Replace \\p{X} and \\P{X} Unicode property escapes for stdlib ``re``.

    Correctly handles \\p{X} both *inside* and *outside* character classes so
    that no nested ``[...]`` are produced.

    Also strips possessive quantifiers (``?+``, ``++``, ``*+``) → greedy.
    """
    import re as _re

    result = []
    i = 0
    n = len(pattern)
    inside_class = False  # are we inside a [...] character class?

    while i < n:
        ch = pattern[i]

        # Track character class boundaries
        if ch == '\\' and i + 1 < n:
            nxt = pattern[i + 1]
            # \p{X} or \P{X}
            if nxt in ('p', 'P') and i + 2 < n and pattern[i + 2] == '{':
                close = pattern.find('}', i + 3)
                if close != -1:
                    prop_name = pattern[i + 3:close]
                    ranges = _UNICODE_RANGES.get(prop_name, None)
                    if ranges is not None:
                        if nxt == 'p':
                            if inside_class:
                                # Inside [...]: just insert ranges directly
                                result.append(ranges)
                            else:
                                # Outside: wrap in [...]
                                result.append('[' + ranges + ']')
                        else:  # \P{X} — negated
                            if inside_class:
                                # Can't negate inside a class easily; skip
                                # (use a placeholder that's unlikely to match)
                                result.append(ranges)
                            else:
                                result.append('[^' + ranges + ']')
                        i = close + 1
                        continue
                    else:
                        # Unknown property, pass through
                        result.append(pattern[i:close + 1])
                        i = close + 1
                        continue
            # Other escape: pass through
            result.append(ch)
            result.append(nxt)
            i += 2
            continue

        if ch == '[' and not inside_class:
            inside_class = True
            result.append(ch)
            # Check for negation: [^
            if i + 1 < n and pattern[i + 1] == '^':
                result.append('^')
                i += 2
                continue
            i += 1
            continue

        if ch == ']' and inside_class:
            inside_class = False
            result.append(ch)
            i += 1
            continue

        result.append(ch)
        i += 1

    converted = ''.join(result)

    # Remove possessive quantifiers (?+, ++, *+) → greedy (?, +, *)
    converted = _re.sub(r'(\?|\+|\*)\+', r'\1', converted)
    return converted


def _compile_pattern(pattern: str):
    """Compile a pre-tokenizer regex pattern.

    Tries the ``regex`` module first (full Unicode property support).
    Falls back to stdlib ``re`` with Unicode property approximations.
    """
    if _HAS_REGEX:
        try:
            return _re_engine.compile(pattern)
        except Exception:
            pass  # Fall through to conversion

    # Convert \p{...} to re-compatible form
    converted = _convert_unicode_props(pattern)
    try:
        return re.compile(converted)
    except re.error:
        # Ultimate fallback: basic GPT-2 style pattern
        return re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
        )
