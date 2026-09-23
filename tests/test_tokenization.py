from src.inference.tokenizer_utils import display_token, raw_token_bytes


def test_display_makes_whitespace_visible_without_stripping():
    assert display_token(" pressure", "Ġpressure") == "␠pressure"
    assert display_token(" ", "Ġ") == "␠"
    assert display_token("\n", "Ċ") == "↵"
    assert display_token("a\tb", "a\tb") == "a⇥b"


def test_partial_utf8_shows_bytes():
    assert raw_token_bytes("Â½") == "½".encode()
    assert display_token("�", "ðŁ") == "<0xF0><0x9F>"


def test_special_tokens_pass_through():
    assert display_token("<|im_start|>", "<|im_start|>", is_special=True) == "<|im_start|>"


def test_trace_token_ids_match_decoded(tokenizer, sampling_trace):
    toks = list(sampling_trace["input_tokens"])
    for s in sampling_trace["steps"]:
        toks += s["top_candidates"]
    for t in toks:
        assert tokenizer.decode([t["token_id"]]) == t["decoded_token"]
        assert tokenizer.convert_ids_to_tokens(t["token_id"]) == t["raw_token"]


def test_trace_input_tokens_reproduce_model_input(tokenizer, sampling_trace):
    ids = [t["token_id"] for t in sampling_trace["input_tokens"]]
    assert tokenizer.decode(ids) == sampling_trace["actual_model_input"]
    shown = [t["token_id"] for t in sampling_trace["input_tokens"] if t["in_display_context"]]
    assert tokenizer.decode(shown) == sampling_trace["display_context"]


def test_tokens_are_not_words(sampling_trace):
    """The DDR context contains sub-word and digit tokens; the trace must keep them as-is."""
    decoded = [t["decoded_token"] for t in sampling_trace["input_tokens"] if t["in_display_context"]]
    assert len(decoded) > len(sampling_trace["display_context"].split())
