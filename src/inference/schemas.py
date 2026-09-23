"""Explicit validation of the inference trace: structure *and* mathematical invariants.

No third-party schema library: every rule is a plain check with a readable error.
"""

import math

# 1.1 added: metadata.tokenizer_vocab_size, token in_tokenizer, step nucleus_complete and appended_text.
SCHEMA_VERSION = "1.1"
TOL = 1e-6

_TOKEN_FIELDS = {
    "token_id": int,
    "raw_token": str,
    "decoded_token": str,
    "display_token": str,
    "is_special": bool,
    "in_tokenizer": bool,
}
_OPT_FLOAT = (float, type(None))
_CANDIDATE_FIELDS = {
    "rank": int,
    **_TOKEN_FIELDS,
    "raw_logit": float,
    "scaled_logit": float,
    "model_probability": float,
    "temperature_probability": float,
    "cumulative_probability": float,
    "inside_top_p": (bool, type(None)),
    "sampling_probability": _OPT_FLOAT,
    "selected": bool,
}
_METADATA_FIELDS = {
    "model": str,
    "model_display_name": str,
    "tokenizer_vocab_size": int,
    "device": str,
    "dtype": str,
    "generation_mode": str,
    "prompt_mode": str,
    "temperature": _OPT_FLOAT,
    "top_p": _OPT_FLOAT,
    "seed": int,
    "steps_requested": int,
    "steps_generated": int,
    "stop_reason": str,
}


class TraceValidationError(ValueError):
    pass


def _check(cond: bool, msg: str):
    if not cond:
        raise TraceValidationError(msg)


def _fields(obj: dict, spec: dict, where: str):
    _check(isinstance(obj, dict), f"{where}: expected object")
    for k, t in spec.items():
        _check(k in obj, f"{where}: missing field {k!r}")
        v = obj[k]
        # bool is a subclass of int; don't let True pass as an int (or an int as a float).
        if t is int:
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif t is float:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        elif t == _OPT_FLOAT:
            ok = v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))
        else:
            ok = isinstance(v, t)
        _check(ok, f"{where}.{k}: bad type {type(v).__name__}")


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isclose(a, b, abs_tol=tol, rel_tol=0)


def _finite(obj: dict, keys, where: str):
    for k in keys:
        v = obj.get(k)
        _check(v is None or math.isfinite(v), f"{where}.{k} is not finite: {v}")


def _log_gap_ok(pa: float, pb: float, expected: float) -> bool:
    """log(pa) - log(pb) must equal the logit gap (softmax identity); skipped when either underflows."""
    if pa < 1e-250 or pb < 1e-250:
        return True
    return abs(math.log(pa) - math.log(pb) - expected) < 1e-6


def validate_trace(trace: dict) -> None:
    version = trace.get("schema_version")
    _check(
        version == SCHEMA_VERSION,
        f"schema_version {version!r} is not supported (expected {SCHEMA_VERSION}); "
        "older traces lack fields added in 1.1, re-capture them with scripts/capture_trace.sh",
    )
    meta = trace.get("metadata")
    _fields(meta, _METADATA_FIELDS, "metadata")
    mode = meta["generation_mode"]
    _check(mode in ("greedy", "sampling"), f"metadata.generation_mode invalid: {mode}")
    _check(meta["prompt_mode"] in ("raw-text", "chat-template"), "metadata.prompt_mode invalid")
    sampling = mode == "sampling"
    _finite(meta, ("temperature", "top_p"), "metadata")
    _check(meta["tokenizer_vocab_size"] > 0, "metadata.tokenizer_vocab_size must be > 0")
    src = trace.get("source_context")
    _check(isinstance(src, dict) and src.get("synthetic") in (True, False, None), "source_context.synthetic must be bool or null")
    if sampling:
        _check(meta["temperature"] is not None and meta["temperature"] > 0, "sampling needs temperature > 0")
        _check(meta["top_p"] is not None and 0 < meta["top_p"] <= 1, "sampling needs 0 < top_p <= 1")
    else:
        _check(meta["temperature"] is None and meta["top_p"] is None, "greedy trace must not claim temperature/top_p")

    for k in ("display_context", "actual_model_input", "generated_text", "final_display_context"):
        _check(isinstance(trace.get(k), str), f"{k} must be a string")
    _check(trace["display_context"] in trace["actual_model_input"], "display_context must appear in actual_model_input")
    _check(
        trace["final_display_context"] == trace["display_context"] + trace["generated_text"],
        "final_display_context must be display_context + generated_text",
    )

    toks = trace.get("input_tokens")
    _check(isinstance(toks, list) and toks, "input_tokens must be a non-empty list")
    for i, t in enumerate(toks):
        _fields(t, {"position": int, **_TOKEN_FIELDS, "in_display_context": bool}, f"input_tokens[{i}]")
        _check(t["position"] == i, f"input_tokens[{i}].position out of order")
    shown = "".join(t["decoded_token"] for t in toks if t["in_display_context"])
    # Concatenated single-token decodes equal the text unless a multi-byte char is split across tokens.
    _check(shown == trace["display_context"] or "�" in shown, "display-context tokens do not reproduce display_context")

    steps = trace.get("steps")
    _check(isinstance(steps, list) and len(steps) == meta["steps_generated"], "steps length != steps_generated")
    prev_after = trace["display_context"]
    appended_so_far = ""
    for s in steps:
        where = f"steps[{s.get('step')}]"
        _check(s["context_before"] == prev_after, f"{where}: context_before must equal previous context_after")
        cands = s["top_candidates"]
        _check(isinstance(cands, list) and cands, f"{where}: top_candidates empty")
        for j, c in enumerate(cands):
            _fields(c, _CANDIDATE_FIELDS, f"{where}.top_candidates[{j}]")
            _check(c["rank"] == j + 1, f"{where}: ranks must be 1..N")
            _finite(
                c, ("raw_logit", "scaled_logit", "model_probability", "temperature_probability", "sampling_probability"), where
            )
            for pk in ("model_probability", "temperature_probability"):
                _check(0 <= c[pk] <= 1 + TOL, f"{where}: {pk} out of [0,1]")
            _check(
                c["in_tokenizer"] == (c["token_id"] < meta["tokenizer_vocab_size"]),
                f"{where}: in_tokenizer inconsistent for id {c['token_id']}",
            )
            # Probabilities must follow from the recorded logits (catches edited probabilities or logits).
            T = meta["temperature"] if sampling else 1.0
            _check(
                math.isclose(c["scaled_logit"], c["raw_logit"] / T, rel_tol=1e-9, abs_tol=1e-9),
                f"{where}: scaled_logit != raw_logit / temperature at rank {c['rank']}",
            )
            if j:
                a = cands[j - 1]
                gap = a["raw_logit"] - c["raw_logit"]
                _check(
                    _log_gap_ok(a["model_probability"], c["model_probability"], gap),
                    f"{where}: model_probability inconsistent with logits at rank {c['rank']}",
                )
                _check(
                    _log_gap_ok(a["temperature_probability"], c["temperature_probability"], gap / T),
                    f"{where}: temperature_probability inconsistent with logits at rank {c['rank']}",
                )

        # Ranking descending (by the distribution top-p is applied to), cumulative sums correct.
        cum = 0.0
        for j, c in enumerate(cands):
            if j:
                _check(
                    c["temperature_probability"] <= cands[j - 1]["temperature_probability"] + 1e-12, f"{where}: not descending"
                )
                _check(
                    c["model_probability"] <= cands[j - 1]["model_probability"] + 1e-9,
                    f"{where}: model_probability order differs",
                )
            cum += c["temperature_probability"]
            _check(_close(c["cumulative_probability"], cum), f"{where}: cumulative_probability wrong at rank {c['rank']}")
        _check(cum <= 1 + TOL, f"{where}: probabilities exceed 1")
        _check(_close(cum + s["omitted_temperature_probability_mass"], 1.0), f"{where}: listed + omitted mass != 1")
        _check(
            _close(sum(c["model_probability"] for c in cands) + s["omitted_model_probability_mass"], 1.0),
            f"{where}: listed + omitted model mass != 1",
        )

        selected = [c for c in cands if c["selected"]]
        _check(len(selected) == 1, f"{where}: exactly one candidate must be selected")
        sel = selected[0]
        st = s["selected_token"]
        for k, v in st.items():
            _check(k in sel and sel[k] == v, f"{where}: selected_token.{k} does not match the selected candidate")

        if sampling:
            p = meta["top_p"]
            n = s["nucleus_size"]
            complete = s["nucleus_complete"]
            _check(
                isinstance(n, int) and n >= 1 and isinstance(complete, bool), f"{where}: nucleus_size/nucleus_complete missing"
            )
            _check(complete == (n <= len(cands)), f"{where}: nucleus_complete inconsistent with exported candidates")
            mass = s["nucleus_temperature_mass"]
            _check(isinstance(mass, float) and 0 < mass <= 1 + TOL, f"{where}: nucleus mass invalid")
            for c in cands:
                before = c["cumulative_probability"] - c["temperature_probability"]
                _check(c["inside_top_p"] == (c["rank"] <= n), f"{where}: nucleus must be a rank prefix")
                if c["rank"] <= n:
                    _check(before < p + TOL, f"{where}: rank {c['rank']} should be outside the nucleus")
                    _check(
                        _close(c["sampling_probability"], c["temperature_probability"] / mass), f"{where}: bad renormalization"
                    )
                else:
                    _check(c["sampling_probability"] == 0.0, f"{where}: excluded token has sampling probability")
            if complete:
                _check(cands[n - 1]["cumulative_probability"] >= p - TOL, f"{where}: nucleus does not reach top_p")
                _check(_close(mass, cands[n - 1]["cumulative_probability"]), f"{where}: nucleus mass wrong")
                total = sum(c["sampling_probability"] for c in cands[:n])
                _check(_close(total, 1.0), f"{where}: sampling probabilities do not sum to 1")
            else:
                # Only a prefix of a very large nucleus is exported: its share must stay below the nucleus total.
                _check(cands[-1]["cumulative_probability"] <= mass + TOL, f"{where}: exported prefix exceeds nucleus mass")
            _check(sel["inside_top_p"] is True, f"{where}: selected token is outside the nucleus")
            if _close(meta["temperature"], 1.0):
                _check(all(_close(c["model_probability"], c["temperature_probability"]) for c in cands), f"{where}: T=1 mismatch")
        else:
            _check(sel["rank"] == 1, f"{where}: greedy must select rank 1")
            _check(s["nucleus_size"] is None and s["nucleus_complete"] is None, f"{where}: greedy has no nucleus")
            for c in cands:
                _check(c["inside_top_p"] is None and c["sampling_probability"] is None, f"{where}: greedy has no nucleus")
                _check(c["model_probability"] == c["temperature_probability"], f"{where}: greedy uses no temperature")

        split_char = "�" in sel["decoded_token"] or "�" in s["context_before"]
        _check(
            s["context_after"] == s["context_before"] + sel["decoded_token"] or split_char,
            f"{where}: context_after must be context_before + selected token",
        )
        # appended_text: what this step added, robust to characters split across tokens.
        _check(isinstance(s.get("appended_text"), str), f"{where}: appended_text missing")
        appended_so_far += s["appended_text"]
        _check(
            s["context_after"].rstrip("\ufffd") == trace["display_context"] + appended_so_far,
            f"{where}: appended_text does not reproduce context_after",
        )
        prev_after = s["context_after"]

    _check(trace["final_display_context"] == prev_after, "final_display_context mismatch")
    _check(trace["generated_text"].rstrip("\ufffd") == appended_so_far, "generated_text does not match the steps")
