import copy

import numpy as np
import pytest

from src.inference.schemas import TraceValidationError, validate_trace
from tests.conftest import GENERATED


def test_generated_traces_validate(sampling_trace, greedy_trace):
    validate_trace(sampling_trace)
    validate_trace(greedy_trace)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t["steps"][0]["top_candidates"][0].__setitem__("model_probability", "0.3"),
        lambda t: t["steps"][0]["top_candidates"][1].__setitem__("cumulative_probability", 0.99),
        lambda t: t["steps"][0]["top_candidates"][0].__setitem__("inside_top_p", False),
        lambda t: t["steps"][0]["top_candidates"][0].__setitem__("sampling_probability", 0.9),
        lambda t: [c.__setitem__("selected", False) for c in t["steps"][0]["top_candidates"]],
        lambda t: t["metadata"].pop("seed"),
        lambda t: t["steps"][1].__setitem__("context_before", "tampered"),
        lambda t: t["steps"][0]["top_candidates"].reverse(),
    ],
)
def test_validator_rejects_tampering(sampling_trace, mutate):
    t = copy.deepcopy(sampling_trace)
    mutate(t)
    with pytest.raises((TraceValidationError, KeyError)):
        validate_trace(t)


def test_validator_rejects_greedy_non_rank_one(greedy_trace):
    t = copy.deepcopy(greedy_trace)
    c = t["steps"][0]["top_candidates"]
    c[0]["selected"], c[1]["selected"] = False, True
    t["steps"][0]["selected_token"].update(rank=2, token_id=c[1]["token_id"])
    with pytest.raises(TraceValidationError):
        validate_trace(t)


def test_json_values_recompute_from_full_logits(sampling_trace):
    """Every exported probability must follow from the saved full-vocabulary logits."""
    npz = GENERATED / "inference_trace.logits.npz"
    if not npz.exists():
        pytest.skip("no logits sidecar")
    z = np.load(npz)
    T = sampling_trace["metadata"]["temperature"]
    assert z["generated_ids"].tolist() == [s["selected_token"]["token_id"] for s in sampling_trace["steps"]]
    for s, raw in zip(sampling_trace["steps"], z["raw_logits"].astype(np.float64)):
        pm = np.exp(raw - raw.max())
        pm /= pm.sum()
        pt = np.exp((raw - raw.max()) / T)
        pt /= pt.sum()
        for c in s["top_candidates"]:
            assert c["raw_logit"] == pytest.approx(raw[c["token_id"]], abs=1e-5)
            assert c["model_probability"] == pytest.approx(pm[c["token_id"]], abs=1e-7)
            assert c["temperature_probability"] == pytest.approx(pt[c["token_id"]], abs=1e-7)
        assert np.argsort(-pt, kind="stable")[0] == s["top_candidates"][0]["token_id"]
