import numpy as np
import pytest

from src.inference.sampling import renormalize, top_p_mask


def test_smallest_set_reaching_p():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    assert top_p_mask(p, 0.8).tolist() == [True, True, False, False]  # 0.5+0.3 = 0.8 reaches p exactly
    assert top_p_mask(p, 0.81).tolist() == [True, True, True, False]
    assert top_p_mask(p, 0.5).tolist() == [True, False, False, False]
    assert top_p_mask(p, 1.0).tolist() == [True, True, True, True]


def test_top_token_always_kept():
    p = np.array([0.95, 0.05])
    assert top_p_mask(p, 0.01).tolist() == [True, False]


def test_unsorted_input():
    p = np.array([0.05, 0.3, 0.15, 0.5])
    assert top_p_mask(p, 0.8).tolist() == [False, True, False, True]


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.1])
def test_invalid_top_p(bad):
    with pytest.raises(ValueError):
        top_p_mask(np.array([1.0]), bad)


def test_renormalize():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    r = renormalize(p, top_p_mask(p, 0.8))
    np.testing.assert_allclose(r, [0.625, 0.375, 0, 0])
    assert r.sum() == pytest.approx(1.0)


def test_trace_nucleus_is_correct(sampling_trace):
    top_p = sampling_trace["metadata"]["top_p"]
    for s in sampling_trace["steps"]:
        cands = s["top_candidates"]
        n = s["nucleus_size"]
        inside = [c for c in cands if c["inside_top_p"]]
        assert len(inside) == n and [c["rank"] for c in inside] == list(range(1, n + 1))
        assert inside[-1]["cumulative_probability"] >= top_p - 1e-9
        if n > 1:
            assert inside[-2]["cumulative_probability"] < top_p  # smallest such set
        assert sum(c["sampling_probability"] for c in inside) == pytest.approx(1.0, abs=1e-9)
        for c in cands:
            if not c["inside_top_p"]:
                assert c["sampling_probability"] == 0.0


def test_trace_matches_huggingface_top_p():
    """Independent check against Hugging Face's own warpers using the saved full-vocabulary logits."""
    torch = pytest.importorskip("torch")
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper

    from tests.conftest import GENERATED, _load

    trace = _load("inference_trace.json")
    npz = GENERATED / "inference_trace.logits.npz"
    if not npz.exists():
        pytest.skip("no logits sidecar")
    logits = np.load(npz)["raw_logits"]
    T, P = trace["metadata"]["temperature"], trace["metadata"]["top_p"]
    for s, raw in zip(trace["steps"], logits):
        x = torch.tensor(raw, dtype=torch.float64)[None]
        x = TopPLogitsWarper(P)(None, TemperatureLogitsWarper(T)(None, x))
        hf = set(torch.nonzero(torch.isfinite(x[0])).flatten().tolist())
        assert hf == {c["token_id"] for c in s["top_candidates"] if c["inside_top_p"]}
