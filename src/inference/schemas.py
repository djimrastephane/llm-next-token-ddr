"""Explicit validation of the inference trace: structure *and* mathematical invariants.

No third-party schema library: every rule is a plain check with a readable error.
"""

import math

SCHEMA_VERSION = "1.0"
TOL = 1e-6

_TOKEN_FIELDS = {"token_id": int, "raw_token": str, "decoded_token": str, "display_token": str, "is_special": bool}
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


def validate_trace(trace: dict) -> None:
    _check(trace.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")
    meta = trace.get("metadata")
    _fields(meta, _METADATA_FIELDS, "metadata")
    mode = meta["generation_mode"]
    _check(mode in ("greedy", "sampling"), f"metadata.generation_mode invalid: {mode}")
    _check(meta["prompt_mode"] in ("raw-text", "chat-template"), "metadata.prompt_mode invalid")
    sampling = mode == "sampling"
    if sampling:
        _check(meta["temperature"] is not None and meta["temperature"] > 0, "sampling needs temperature > 0")
        _check(meta["top_p"] is not None and 0 < meta["top_p"] <= 1, "sampling needs 0 < top_p <= 1")
    else:
        _check(meta["temperature"] is None and meta["top_p"] is None, "greedy trace must not claim temperature/top_p")

    for k in ("display_context", "actual_model_input", "generated_text", "final_display_context"):
        _check(isinstance(trace.get(k), str), f"{k} must be a string")
    _check(trace["display_context"] in trace["actual_model_input"], "display_context must appear in actual_model_input")

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
    for s in steps:
        where = f"steps[{s.get('step')}]"
        _check(s["context_before"] == prev_after, f"{where}: context_before must equal previous context_after")
        cands = s["top_candidates"]
        _check(isinstance(cands, list) and cands, f"{where}: top_candidates empty")
        for j, c in enumerate(cands):
            _fields(c, _CANDIDATE_FIELDS, f"{where}.top_candidates[{j}]")
            _check(c["rank"] == j + 1, f"{where}: ranks must be 1..N")
            for pk in ("model_probability", "temperature_probability"):
                _check(0 <= c[pk] <= 1 + TOL, f"{where}: {pk} out of [0,1]")

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
        _check(st["token_id"] == sel["token_id"] and st["rank"] == sel["rank"], f"{where}: selected_token mismatch")

        if sampling:
            p = meta["top_p"]
            n = s["nucleus_size"]
            _check(isinstance(n, int) and 1 <= n <= len(cands), f"{where}: full nucleus must be exported")
            for c in cands:
                before = c["cumulative_probability"] - c["temperature_probability"]
                _check(c["inside_top_p"] == (c["rank"] <= n), f"{where}: nucleus must be a rank prefix")
                if c["rank"] <= n:
                    _check(before < p + TOL, f"{where}: rank {c['rank']} should be outside the nucleus")
                else:
                    _check(c["sampling_probability"] == 0.0, f"{where}: excluded token has sampling probability")
            _check(cands[n - 1]["cumulative_probability"] >= p - TOL, f"{where}: nucleus does not reach top_p")
            mass = s["nucleus_temperature_mass"]
            _check(_close(mass, cands[n - 1]["cumulative_probability"]), f"{where}: nucleus mass wrong")
            total = 0.0
            for c in cands[:n]:
                _check(_close(c["sampling_probability"], c["temperature_probability"] / mass), f"{where}: bad renormalization")
                total += c["sampling_probability"]
            _check(_close(total, 1.0), f"{where}: sampling probabilities do not sum to 1")
            _check(sel["inside_top_p"] is True, f"{where}: selected token is outside the nucleus")
            if _close(meta["temperature"], 1.0):
                _check(all(_close(c["model_probability"], c["temperature_probability"]) for c in cands), f"{where}: T=1 mismatch")
        else:
            _check(sel["rank"] == 1, f"{where}: greedy must select rank 1")
            for c in cands:
                _check(c["inside_top_p"] is None and c["sampling_probability"] is None, f"{where}: greedy has no nucleus")
                _check(c["model_probability"] == c["temperature_probability"], f"{where}: greedy uses no temperature")

        split_char = "�" in sel["decoded_token"] or "�" in s["context_before"]
        _check(
            s["context_after"] == s["context_before"] + sel["decoded_token"] or split_char,
            f"{where}: context_after must be context_before + selected token",
        )
        prev_after = s["context_after"]

    _check(trace["final_display_context"] == prev_after, "final_display_context mismatch")
