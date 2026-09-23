import numpy as np
import pytest

from src.inference.sampling import decide


def test_greedy_selects_argmax_and_ignores_rng():
    logits = np.array([0.1, 5.0, 4.9, -2.0])
    d = decide(logits, "greedy")
    assert d.selected_id == 1 and d.nucleus is None and d.sampling_probs is None


def test_sampling_selects_only_inside_nucleus():
    rng = np.random.default_rng(0)
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    for _ in range(2000):
        d = decide(logits, "sampling", temperature=1.0, top_p=0.8, rng=rng)
        assert d.nucleus[d.selected_id]


def test_sampling_frequencies_follow_renormalized_probs():
    rng = np.random.default_rng(123)
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    picks = [decide(logits, "sampling", temperature=1.0, top_p=0.75, rng=rng).selected_id for _ in range(20000)]
    freq = np.bincount(picks, minlength=4) / len(picks)
    np.testing.assert_allclose(freq, [0.625, 0.375, 0, 0], atol=0.015)


def test_sampling_can_select_non_top_token():
    rng = np.random.default_rng(0)
    logits = np.log(np.array([0.4, 0.35, 0.25]))
    picks = {decide(logits, "sampling", temperature=1.0, top_p=1.0, rng=rng).selected_id for _ in range(200)}
    assert picks == {0, 1, 2}


def test_same_seed_same_selection():
    logits = np.random.default_rng(5).normal(size=500)
    a = [decide(logits, "sampling", 0.7, 0.9, np.random.default_rng(42)).selected_id for _ in range(3)]
    assert len(set(a)) == 1


def test_unknown_mode():
    with pytest.raises(ValueError):
        decide(np.zeros(3), "beam")


def test_trace_greedy_always_rank_one(greedy_trace):
    assert greedy_trace["metadata"]["generation_mode"] == "greedy"
    for s in greedy_trace["steps"]:
        assert s["selected_token"]["rank"] == 1


def test_trace_sampled_token_in_nucleus(sampling_trace):
    for s in sampling_trace["steps"]:
        sel = [c for c in s["top_candidates"] if c["selected"]]
        assert len(sel) == 1 and sel[0]["inside_top_p"] is True
        assert s["selected_token"]["token_id"] == sel[0]["token_id"]


def test_greedy_and_sampling_share_starting_context(sampling_trace, greedy_trace):
    assert sampling_trace["display_context"] == greedy_trace["display_context"]
    assert sampling_trace["input_tokens"] == greedy_trace["input_tokens"]
    # Step 1 sees the same context, so the model's own distribution must be the same.
    a = {c["token_id"]: c["model_probability"] for c in sampling_trace["steps"][0]["top_candidates"]}
    b = {c["token_id"]: c["model_probability"] for c in greedy_trace["steps"][0]["top_candidates"]}
    for k in set(a) & set(b):
        assert a[k] == pytest.approx(b[k], abs=1e-4)


def test_ends_sentence_rule():
    from src.inference.capture_trace import ends_sentence

    assert ends_sentence(" constant at 3,800 psi.") and ends_sentence(" stable!") and ends_sentence(" ok\n")
    assert not ends_sentence(" psi") and not ends_sentence(",") and not ends_sentence(" 3")


def test_abbreviation_or_decimal_period_does_not_stop():
    from src.inference.capture_trace import ends_sentence

    assert not ends_sentence(" drilled with 8½ in.")  # tokenizer emits " in" then "." separately
    assert not ends_sentence(" to 12,450 ft.")
    assert not ends_sentence(" approx.")
    assert not ends_sentence(" at 3.")  # possible decimal point


def test_trace_stops_exactly_at_sentence_end(sampling_trace):
    from src.inference.capture_trace import ends_sentence

    if sampling_trace["metadata"].get("until_sentence_end"):
        prompt = sampling_trace["display_context"]
        texts = [s["context_after"][len(prompt) :] for s in sampling_trace["steps"]]
        ends = [ends_sentence(t, prompt) for t in texts]
        assert ends[-1] == (sampling_trace["metadata"]["stop_reason"] == "sentence_end")
        assert not any(ends[:-1])  # nothing ended the sentence earlier
