import numpy as np

from src.inference.sampling import rank_order, softmax


def test_softmax_sums_to_one():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = softmax(rng.normal(0, 10, size=5000))
        assert abs(p.sum() - 1.0) < 1e-12
        assert (p >= 0).all()


def test_softmax_is_shift_invariant_and_stable():
    x = np.array([1000.0, 999.0, 998.0])  # would overflow a naive exp
    p = softmax(x)
    np.testing.assert_allclose(p, softmax(x - 1000.0), atol=1e-15)
    assert np.isfinite(p).all()


def test_rank_order_descending_with_stable_ties():
    p = np.array([0.1, 0.4, 0.1, 0.4])
    assert rank_order(p).tolist() == [1, 3, 0, 2]


def test_trace_listed_plus_omitted_mass_is_one(sampling_trace):
    for s in sampling_trace["steps"]:
        listed = sum(c["model_probability"] for c in s["top_candidates"])
        assert abs(listed + s["omitted_model_probability_mass"] - 1.0) < 1e-6


def test_trace_ranking_descending(sampling_trace, greedy_trace):
    for t in (sampling_trace, greedy_trace):
        for s in t["steps"]:
            probs = [c["temperature_probability"] for c in s["top_candidates"]]
            assert probs == sorted(probs, reverse=True)
            assert [c["rank"] for c in s["top_candidates"]] == list(range(1, len(probs) + 1))
