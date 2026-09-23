"""Token representations.

Every token is described three ways:
  raw_token      - the tokenizer's internal vocabulary string (e.g. "Ġpressure" in byte-level BPE)
  decoded_token  - the exact text the token decodes to on its own (e.g. " pressure"); never stripped
  display_token  - a UI-safe rendering where invisible characters are made visible (e.g. "␠pressure")
"""

from functools import lru_cache

# Characters that would be invisible or break layout in the video.
_VISIBLE = {
    " ": "␠",
    "\n": "↵",
    "\r": "␍",
    "\t": "⇥",
}


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode mapping used by byte-level BPE tokenizers (Qwen, GPT-2, Llama 3)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def raw_token_bytes(raw_token: str) -> bytes | None:
    """Bytes of a byte-level BPE token, or None if the token is not in byte-level form."""
    dec = _byte_decoder()
    if not all(ch in dec for ch in raw_token):
        return None
    return bytes(dec[ch] for ch in raw_token)


def display_token(decoded: str, raw_token: str, is_special: bool = False) -> str:
    """UI-safe rendering. Leading/trailing whitespace is made visible, never removed."""
    if is_special:
        return decoded
    if "\ufffd" in decoded:
        # The token is a fragment of a multi-byte UTF-8 character; show its bytes instead.
        b = raw_token_bytes(raw_token)
        if b is not None:
            return "".join(f"<0x{x:02X}>" for x in b)
        return decoded
    if decoded == "":
        return "∅"
    out = "".join(_VISIBLE.get(ch, ch) for ch in decoded)
    # Other control characters: show code point.
    return "".join(ch if ch.isprintable() or ch in _VISIBLE.values() else f"<U+{ord(ch):04X}>" for ch in out)


_SPECIAL_CACHE: dict[int, frozenset[int]] = {}


def _special_ids(tokenizer) -> frozenset[int]:
    key = id(tokenizer)
    if key not in _SPECIAL_CACHE:
        _SPECIAL_CACHE[key] = frozenset(tokenizer.all_special_ids) | frozenset(tokenizer.get_added_vocab().values())
    return _SPECIAL_CACHE[key]


def describe_token(tokenizer, token_id: int) -> dict:
    raw = tokenizer.convert_ids_to_tokens(token_id)
    decoded = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    is_special = token_id in _special_ids(tokenizer)
    return {
        "token_id": int(token_id),
        "raw_token": raw,
        "decoded_token": decoded,
        "display_token": display_token(decoded, raw, is_special),
        "is_special": bool(is_special),
    }
