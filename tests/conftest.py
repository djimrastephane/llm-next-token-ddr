import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "data" / "generated"
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _load(name):
    p = GENERATED / name
    if not p.exists():
        pytest.skip(f"{p} not generated yet; run scripts/capture_trace.sh")
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sampling_trace():
    return _load("inference_trace.json")


@pytest.fixture(scope="session")
def greedy_trace():
    return _load("inference_trace_greedy.json")


@pytest.fixture(scope="session")
def tokenizer(sampling_trace):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(sampling_trace["metadata"]["model"])
    except OSError as e:  # model not in local cache
        pytest.skip(f"tokenizer unavailable offline: {e}")
