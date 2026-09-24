"""Regression tests for the second-pass QA/QC + security review (IDs R1, R2, F1-F8 of that review)."""

import copy
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.inference.capture_trace import ends_sentence, export_count, load_context, parse_args
from src.inference.sampling import apply_temperature, decide
from src.inference.schemas import TraceValidationError, validate_trace
from src.inference.tokenizer_utils import describe_token, incremental_texts
from tests.conftest import GENERATED, ROOT


def _rejects(trace, mutate, match=None):
    t = copy.deepcopy(trace)
    mutate(t)
    with pytest.raises(TraceValidationError, match=match):
        validate_trace(t)


# R1: schema change is explicit; old traces are rejected with a clear message, not a crash.
def test_r1_schema_1_0_trace_rejected_clearly(sampling_trace):
    def to_v1_0(t):
        t["schema_version"] = "1.0"
        del t["metadata"]["tokenizer_vocab_size"]

    _rejects(sampling_trace, to_v1_0, match="not supported .*re-capture")


# R2: the word before a generated period may come from the prompt.
@pytest.mark.parametrize(
    "prompt, generated, expected",
    [
        ("Drilled 8½ in", ".", False),  # abbreviation split across prompt/generated
        ("Pump pressure 3", ".", False),  # possible decimal point
        ("Pump pressure remained constant", ".", True),
        ("Line one.\nPump pressure remained", " at", False),  # newline in the prompt must not stop generation
        ("Pump pressure remained", " constant\n", True),
    ],
)
def test_r2_prompt_boundary(prompt, generated, expected):
    assert ends_sentence(generated, prompt) is expected


# F1: output rows beyond the tokenizer are described, not crashed on.
def test_f1_padding_row_described(tokenizer):
    pad_id = len(tokenizer)  # first output row with no token (Qwen: 151665)
    d = describe_token(tokenizer, pad_id)
    assert d["in_tokenizer"] is False and d["raw_token"] == "" and d["decoded_token"] == ""
    assert str(pad_id) in d["display_token"]
    assert describe_token(tokenizer, 7262)["in_tokenizer"] is True


def test_f1_export_count_caps_huge_nucleus_but_keeps_selection():
    assert export_count(10, 7, 2, 1000) == 10
    assert export_count(10, 151936, 3, 1000) == 1000  # top-p = 1: cap, marked incomplete in the trace
    assert export_count(10, 151936, 1852, 1000) == 1852  # selected token always exported


def test_f1_top_p_one_nucleus_includes_padding_rows(tokenizer):
    raw = np.load(GENERATED / "inference_trace.logits.npz")["raw_logits"][0]
    d = decide(raw, "sampling", 0.7, 1.0, np.random.default_rng(0))
    assert d.nucleus.sum() == raw.shape[0] > len(tokenizer)  # math stays faithful to the full output layer


# F2: integrity checks catch edits that the old validator accepted.
@pytest.mark.parametrize(
    "name, mutate",
    [
        ("fabricated generated_text", lambda t: t.__setitem__("generated_text", " stable at 9,999 psi.")),
        ("selected model_probability", lambda t: t["steps"][0]["selected_token"].__setitem__("model_probability", 0.99)),
        ("selected display_token", lambda t: t["steps"][0]["selected_token"].__setitem__("display_token", "␠stable")),
        ("non-finite logit", lambda t: t["steps"][0]["top_candidates"][0].__setitem__("raw_logit", math.inf)),
        ("probability not from logits", lambda t: t["steps"][0]["top_candidates"][1].__setitem__("model_probability", 0.2)),
        ("appended_text edited", lambda t: t["steps"][0].__setitem__("appended_text", " stable")),
        ("provenance type", lambda t: t["source_context"].__setitem__("synthetic", "yes")),
    ],
)
def test_f2_tampering_rejected(sampling_trace, name, mutate):
    _rejects(sampling_trace, mutate)


# F4: a character split across tokens is credited to the token that completes it.
def test_f4_split_unicode(tokenizer):
    ids = tokenizer("🛢", add_special_tokens=False).input_ids
    assert len(ids) > 1  # the emoji really is split
    pieces = incremental_texts(tokenizer, ids)
    assert "".join(pieces) == "🛢" and pieces[-1] == "🛢" and all(p == "" for p in pieces[:-1])


def test_f4_trace_appended_text_reproduces_generation(sampling_trace, greedy_trace):
    for t in (sampling_trace, greedy_trace):
        assert "".join(s["appended_text"] for s in t["steps"]) == t["generated_text"]


# F5: provenance is recorded, never assumed.
def test_f5_custom_context_has_unknown_provenance(tmp_path):
    args = parse_args(["--context", "Pump pressure remained"])
    assert load_context(args)["synthetic"] is None
    f = tmp_path / "ctx.json"
    f.write_text(json.dumps({"default_context_id": "a", "contexts": [{"id": "a", "text": "x", "facts": {}}]}))
    assert load_context(parse_args(["--contexts-file", str(f)]))["synthetic"] is None
    assert load_context(parse_args([]))["synthetic"] is True  # repo file declares "synthetic": true


# F6: the wrapper writes two files, so a single --output must be refused.
@pytest.mark.parametrize("arg", [["--output", "x.json"], ["--output=x.json"]])
def test_f6_wrapper_rejects_output(arg):
    r = subprocess.run([str(ROOT / "scripts" / "capture_trace.sh"), *arg], capture_output=True, text=True)
    assert r.returncode == 2 and "OUT_DIR" in r.stderr


# F7: documented Node version matches the locked toolchain.
def test_f7_node_requirement_documented():
    engines = json.loads((ROOT / "package.json").read_text())["engines"]["node"]
    assert engines == ">=22.12"
    assert "Node 22.12+" in (ROOT / "README.md").read_text(encoding="utf-8")


# F8: non-finite temperature / top-p rejected at the CLI and in the maths.
@pytest.mark.parametrize("flag, value", [("--temperature", "nan"), ("--temperature", "inf"), ("--top-p", "nan")])
def test_f8_cli_rejects_non_finite(flag, value):
    with pytest.raises(SystemExit):
        parse_args([flag, value])


@pytest.mark.parametrize("T", [math.inf, math.nan])
def test_f8_apply_temperature_rejects_non_finite(T):
    with pytest.raises(ValueError, match="finite"):
        apply_temperature(np.array([1.0, 2.0]), T)


# Integration: a top-p = 1 capture (nucleus = whole output layer) completes and validates. Needs cached weights.
@pytest.mark.slow
def test_f1_capture_top_p_one_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from src.inference.capture_trace import main

    out = tmp_path / "t.json"
    try:
        trace = main(
            ["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--top-p", "1.0", "--steps", "2", "--output", str(out), "--no-save-logits"]
        )
    except OSError as e:
        pytest.skip(f"model not cached: {e}")
    validate_trace(json.loads(Path(out).read_text()))
    assert all(s["nucleus_complete"] is False for s in trace["steps"])


# Hardening: the default model is loaded at a pinned commit; other models default to their latest revision.
def test_default_model_revision_pinned():
    from src.inference.capture_trace import DEFAULT_MODEL, PINNED_REVISIONS

    assert parse_args([]).revision == PINNED_REVISIONS[DEFAULT_MODEL]
    assert parse_args(["--model", "Qwen/Qwen2.5-0.5B-Instruct"]).revision is None
    assert parse_args(["--revision", "abc123"]).revision == "abc123"


def test_committed_trace_matches_pinned_revision(sampling_trace):
    from src.inference.capture_trace import PINNED_REVISIONS

    m = sampling_trace["metadata"]
    assert m["model_revision"] == PINNED_REVISIONS[m["model"]]
