import numpy as np
import pytest

from src.inference.capture_trace import parse_args
from src.inference.sampling import apply_temperature, decide, softmax


def test_temperature_divides_logits():
    np.testing.assert_allclose(apply_temperature(np.array([2.0, 1.0]), 0.5), [4.0, 2.0])


@pytest.mark.parametrize("t", [0, 0.0, -0.5])
def test_zero_or_negative_temperature_raises_not_divides(t):
    with np.errstate(divide="raise", invalid="raise"):
        with pytest.raises(ValueError, match="greedy"):
            apply_temperature(np.array([1.0, 2.0]), t)


def test_cli_rejects_temperature_zero_for_sampling():
    with pytest.raises(SystemExit):
        parse_args(["--mode", "sampling", "--temperature", "0"])


def test_cli_greedy_ignores_temperature_zero():
    args = parse_args(["--mode", "greedy", "--temperature", "0"])
    assert args.mode == "greedy"


def test_lower_temperature_sharpens_higher_flattens():
    logits = np.array([3.0, 2.0, 1.0, 0.0])
    p1 = softmax(logits)
    assert softmax(logits / 0.5)[0] > p1[0] > softmax(logits / 2.0)[0]


def test_temperature_preserves_ranking():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=1000)
    d = decide(logits, "sampling", temperature=0.3, top_p=1.0, rng=np.random.default_rng(0))
    assert d.order.tolist() == np.argsort(-logits, kind="stable").tolist()


def test_trace_temperature_probability_recomputable(sampling_trace):
    """temperature_probability must equal softmax(raw/T) relative to the model probability (same normalizer check
    via ratios between listed candidates)."""
    T = sampling_trace["metadata"]["temperature"]
    for s in sampling_trace["steps"]:
        c = s["top_candidates"]
        for a, b in zip(c, c[1:]):
            if b["temperature_probability"] > 1e-12:
                ratio = a["temperature_probability"] / b["temperature_probability"]
                expected = np.exp((a["raw_logit"] - b["raw_logit"]) / T)
                assert ratio == pytest.approx(expected, rel=1e-4)
            assert a["scaled_logit"] == pytest.approx(a["raw_logit"] / T, rel=1e-9)


def test_temperature_underflow_rejected_not_nan():
    with pytest.raises(ValueError, match="too small"):
        apply_temperature(np.array([1.0, 2.0]), 1e-320)
    with pytest.raises(ValueError):
        decide(np.array([1.0, 2.0]), "sampling", 1e-320, 0.9, np.random.default_rng(0))
