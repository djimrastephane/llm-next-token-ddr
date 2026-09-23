"""The video must read candidate tokens and probabilities from the trace, never from literals."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "src" / "video"
LOADER = VIDEO / "data" / "loadInferenceTrace.ts"

# e.g. `probability: 0.58` or `{ token: "pressure"` inside a component
HARDCODED = re.compile(r"(probability|prob|logit)\s*:\s*-?\d*\.\d+|token\s*:\s*['\"]", re.I)


def _tsx_files():
    return [p for p in VIDEO.rglob("*.ts*") if "node_modules" not in p.parts and not p.name.endswith(".test.ts")]


def test_loader_imports_generated_trace():
    src = LOADER.read_text(encoding="utf-8")
    assert "data/generated/inference_trace" in src


def test_no_hardcoded_candidates_in_video_code():
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in _tsx_files()
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if HARDCODED.search(line)
    ]
    assert not offenders, offenders


def test_only_loader_imports_trace_json():
    imp = re.compile(r"^\s*import .*inference_trace.*\.json", re.M)
    importers = [p for p in _tsx_files() if imp.search(p.read_text(encoding="utf-8")) and p != LOADER]
    assert not importers, f"trace must be accessed via loadInferenceTrace.ts only: {importers}"


def test_video_disclaims_generated_numbers():
    """Generated text can contain engineering numbers; the video must say they are not calculations."""
    final = (VIDEO / "scenes" / "S10Final.tsx").read_text(encoding="utf-8")
    assert "NOT A HYDRAULICS CALCULATION OR MEASUREMENT" in final
    assert "not a calculation" in (VIDEO / "scenes" / "S8Loop.tsx").read_text(encoding="utf-8")
