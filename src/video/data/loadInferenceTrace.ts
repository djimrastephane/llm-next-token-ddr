// The ONLY place the video reads model output. Every token, probability, rank and selection
// shown on screen comes from the JSON trace captured by src/inference/capture_trace.py.
// Components receive values from here; they never define candidate data themselves.

import sampledTraceJson from "../../../data/generated/inference_trace.json";
import type { Candidate, InferenceTrace, Step } from "./types";

const TOL = 1e-6;
export const SCHEMA_VERSION = "1.1";

function fail(msg: string): never {
  throw new Error(`[inference trace] ${msg}. Re-run scripts/capture_trace.sh.`);
}

const isProb = (x: unknown) => typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 1 + TOL;
const isNum = (x: unknown) => typeof x === "number" && Number.isFinite(x);

/** Validate the invariants the animation relies on, then return the typed trace. */
export function loadInferenceTrace(json: unknown): InferenceTrace {
  const t = json as InferenceTrace;
  if (t?.schema_version !== SCHEMA_VERSION) fail(`unsupported schema_version ${t?.schema_version} (expected ${SCHEMA_VERSION})`);
  const m = t.metadata;
  if (!Number.isInteger(m?.tokenizer_vocab_size) || m.tokenizer_vocab_size <= 0) fail("metadata.tokenizer_vocab_size missing");
  if (!t.steps?.length) fail("no generation steps");
  const sampling = m.generation_mode === "sampling";
  if (sampling) {
    if (!isNum(m.temperature) || m.temperature! <= 0) fail("sampling needs a finite temperature > 0");
    if (!isNum(m.top_p) || m.top_p! <= 0 || m.top_p! > 1) fail("sampling needs 0 < top_p <= 1");
  }
  if (![true, false, null].includes(t.source_context?.synthetic as boolean | null)) fail("source_context.synthetic must be bool or null");
  let appended = "";
  for (const s of t.steps) {
    const c = s.top_candidates;
    if (!c.length) fail(`step ${s.step} has no candidates`);
    c.forEach((x, i) => {
      if (x.rank !== i + 1) fail(`step ${s.step}: ranks out of order`);
      if (!isProb(x.model_probability) || !isProb(x.temperature_probability) || !isNum(x.raw_logit))
        fail(`step ${s.step}: invalid number at rank ${x.rank}`);
      if (x.sampling_probability !== null && !isProb(x.sampling_probability)) fail(`step ${s.step}: invalid sampling probability`);
      if (i > 0 && x.temperature_probability > c[i - 1].temperature_probability + 1e-12) fail(`step ${s.step}: not descending`);
    });
    const sel = c.filter((x) => x.selected);
    if (sel.length !== 1) fail(`step ${s.step}: exactly one candidate must be selected`);
    for (const [k, v] of Object.entries(s.selected_token)) {
      if ((sel[0] as Record<string, unknown>)[k] !== v) fail(`step ${s.step}: selected_token.${k} contradicts the selected candidate`);
    }
    if (sampling) {
      const inside = c.filter((x) => x.inside_top_p);
      if (!sel[0].inside_top_p) fail(`step ${s.step}: selected token outside nucleus`);
      if (s.nucleus_complete) {
        const sum = inside.reduce((a, x) => a + (x.sampling_probability ?? 0), 0);
        if (inside.length !== s.nucleus_size || Math.abs(sum - 1) > TOL) fail(`step ${s.step}: nucleus not fully exported`);
      } else if (s.nucleus_complete !== false || inside.length > (s.nucleus_size ?? 0)) {
        fail(`step ${s.step}: nucleus_complete inconsistent`);
      }
    } else if (sel[0].rank !== 1) {
      fail(`step ${s.step}: greedy must select rank 1`);
    }
    if (typeof s.appended_text !== "string") fail(`step ${s.step}: appended_text missing`);
    appended += s.appended_text;
  }
  if (appended !== t.generated_text.replace(/\uFFFD+$/, "")) fail("appended_text does not reproduce generated_text");
  return t;
}

export const trace: InferenceTrace = loadInferenceTrace(sampledTraceJson);

if (trace.metadata.generation_mode !== "sampling") {
  // The main video explains temperature and top-p, which only exist in sampling mode.
  fail("the NextTokenDDR video needs a sampling trace in data/generated/inference_trace.json");
}

export const meta = trace.metadata;
export const firstStep: Step = trace.steps[0];

/**
 * The first k candidates for display. If the selected one ranks lower than that, it replaces the k-th bar
 * (its rank label shows the gap), so a selection is never hidden and the chart never grows past k bars.
 */
export function withSelected(cands: Candidate[], k: number): Candidate[] {
  const head = cands.slice(0, k);
  if (head.some((c) => c.selected)) return head;
  const sel = cands.find((c) => c.selected);
  return sel ? [...cands.slice(0, k - 1), sel] : head;
}

/** Top-ranked candidates of a step for display, always including the selected token. */
export const displayCandidates = (step: Step, k: number): Candidate[] => withSelected(step.top_candidates, k);

/** Bars shown for a fast autoregressive step: the nucleus plus two excluded tokens, at most 8. */
export const loopCandidates = (step: Step): Candidate[] => displayCandidates(step, Math.min(8, (step.nucleus_size ?? 1) + 2));

/** Text each generated step appended to the context (from the trace, robust to split characters). */
export const appendedTexts: string[] = trace.steps.map((s) => s.appended_text);

/** How to label the input text: never claims "synthetic" unless the trace says so. */
export function provenanceFor(synthetic: boolean | null): { label: string; sentence: string } {
  if (synthetic === true) return { label: "SYNTHETIC EXAMPLE", sentence: "Synthetic DDR example." };
  if (synthetic === false) return { label: "REPORT TEXT", sentence: "DDR text." };
  return { label: "CUSTOM INPUT · PROVENANCE NOT RECORDED", sentence: "Custom input (provenance not recorded)." };
}

export const provenance = provenanceFor(trace.source_context.synthetic);

/** Tokens of the DDR text itself (excludes any chat-template prefix). */
export const ddrTokens = trace.input_tokens.filter((t) => t.in_display_context);

/** Oil & Gas facts stated in the DDR text (NOT model parameters). */
export const ddrFacts = trace.source_context.facts;
