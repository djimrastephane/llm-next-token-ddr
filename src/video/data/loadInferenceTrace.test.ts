import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { displayCandidates, firstStep, loadInferenceTrace, trace } from "./loadInferenceTrace";

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
