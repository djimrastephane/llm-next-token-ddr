import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { appendedTexts, comparison, displayCandidates, loadComparison, firstStep, loadInferenceTrace, provenance, provenanceFor, trace, withSelected } from "./loadInferenceTrace";

const file = JSON.parse(readFileSync("data/generated/inference_trace.json", "utf-8"));

describe("loadInferenceTrace", () => {
  it("serves exactly the Python trace (no substituted values)", () => {
    expect(trace).toEqual(file);
  });

  it("display candidates are the trace's top-ranked candidates, unchanged", () => {
    for (const s of trace.steps) {
      const shown = displayCandidates(s, 8);
      expect(shown).toEqual(s.top_candidates.slice(0, shown.length));
      expect(shown.some((c) => c.selected)).toBe(true);
    }
  });

  it("selected token shown in the video matches the trace", () => {
    expect(firstStep.selected_token.token_id).toBe(file.steps[0].selected_token.token_id);
    expect(trace.generated_text).toBe(file.generated_text);
  });

  it("rejects tampered traces", () => {
    const bad = structuredClone(file);
    bad.steps[0].top_candidates[0].selected = !bad.steps[0].top_candidates[0].selected;
    expect(() => loadInferenceTrace(bad)).toThrow();
    const bad2 = structuredClone(file);
    bad2.steps[0].top_candidates.reverse();
    expect(() => loadInferenceTrace(bad2)).toThrow();
  });
});

// Second-pass review regressions (R1, F2, F3, F4, F5).
describe("review regressions", () => {
  it("R1: rejects a schema-1.0 trace up front instead of crashing in a scene", () => {
    const old = structuredClone(file);
    old.schema_version = "1.0";
    delete old.metadata.tokenizer_vocab_size;
    expect(() => loadInferenceTrace(old)).toThrow(/schema_version/);
  });

  it("F2: rejects contradictory selected token, negative probability, negative temperature", () => {
    const a = structuredClone(file);
    a.steps[0].selected_token.display_token = "␠tampered";
    expect(() => loadInferenceTrace(a)).toThrow(/contradicts/);
    const b = structuredClone(file);
    b.steps[0].top_candidates[3].model_probability = -0.1;
    expect(() => loadInferenceTrace(b)).toThrow();
    const c = structuredClone(file);
    c.metadata.temperature = -0.7;
    expect(() => loadInferenceTrace(c)).toThrow(/temperature/);
  });

  it("F3: a selection ranked below the bar limit stays visible", () => {
    const cands = Array.from({ length: 12 }, (_, i) => ({ ...file.steps[0].top_candidates[0], rank: i + 1, token_id: i, selected: i === 8 }));
    const shown = withSelected(cands, 8);
    expect(shown).toHaveLength(8);
    expect(shown.findIndex((c) => c.selected)).toBe(7);
    expect(shown[7].rank).toBe(9);
    expect(withSelected(cands, 10)).toEqual(cands.slice(0, 10));
  });

  it("F4: context growth comes from appended_text, which reproduces the generated text", () => {
    expect(appendedTexts.join("")).toBe(trace.generated_text);
    const bad = structuredClone(file);
    bad.steps[0].appended_text = " stable";
    expect(() => loadInferenceTrace(bad)).toThrow(/appended_text/);
  });

  it("F5: provenance label follows the trace, never assumed", () => {
    expect(provenance).toEqual(provenanceFor(file.source_context.synthetic));
    expect(provenanceFor(true).label).toBe("SYNTHETIC EXAMPLE");
    expect(provenanceFor(null).label).toMatch(/CUSTOM INPUT/);
    expect(provenanceFor(null).sentence).not.toMatch(/Synthetic/);
    expect(provenanceFor(false).label).not.toMatch(/SYNTHETIC/);
    const custom = structuredClone(file);
    custom.source_context.synthetic = "yes";
    expect(() => loadInferenceTrace(custom)).toThrow(/synthetic/);
  });
});

describe("greedy vs sampling comparison", () => {
  const greedyFile = JSON.parse(readFileSync("data/generated/inference_trace_greedy.json", "utf-8"));

  it("pairs the committed traces and finds the first differing step from the data", () => {
    const expected = file.steps.findIndex(
      (s: { selected_token: { token_id: number } }, i: number) => s.selected_token.token_id !== greedyFile.steps[i]?.selected_token.token_id,
    );
    expect(comparison.diverge).toBe(expected === -1 ? null : expected);
    expect(comparison.greedy.generated_text).toBe(greedyFile.generated_text);
    expect(comparison.greedy.steps.every((s) => s.selected_token.rank === 1)).toBe(true);
  });

  it("refuses to compare runs from different contexts or models", () => {
    const otherContext = structuredClone(greedyFile);
    otherContext.display_context = "Something else";
    expect(() => loadComparison(trace, otherContext)).toThrow();
    const otherModel = structuredClone(greedyFile);
    otherModel.metadata.model = "another/model";
    expect(() => loadComparison(trace, otherModel)).toThrow(/differ/);
    expect(() => loadComparison(trace, structuredClone(file))).toThrow(/not a greedy trace/);
  });
});
