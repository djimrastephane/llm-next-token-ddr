// The ONLY place the video reads model output. Every token, probability, rank and selection
// shown on screen comes from the JSON trace captured by src/inference/capture_trace.py.
// Components receive values from here; they never define candidate data themselves.

import sampledTraceJson from "../../../data/generated/inference_trace.json";
import type { Candidate, InferenceTrace, Step } from "./types";

const TOL = 1e-6;

function fail(msg: string): never {
  throw new Error(`[inference trace] ${msg}. Re-run scripts/capture_trace.sh.`);
}

/** Validate the invariants the animation relies on, then return the typed trace. */
export function loadInferenceTrace(json: unknown): InferenceTrace {
  const t = json as InferenceTrace;
  if (t?.schema_version !== "1.0") fail(`unsupported schema_version ${t?.schema_version}`);
  if (!t.steps?.length) fail("no generation steps");
  const sampling = t.metadata.generation_mode === "sampling";
  for (const s of t.steps) {
    const c = s.top_candidates;
    if (!c.length) fail(`step ${s.step} has no candidates`);
    c.forEach((x, i) => {
      if (x.rank !== i + 1) fail(`step ${s.step}: ranks out of order`);
      if (i > 0 && x.temperature_probability > c[i - 1].temperature_probability + 1e-12) fail(`step ${s.step}: not descending`);
    });
    const sel = c.filter((x) => x.selected);
    if (sel.length !== 1 || sel[0].token_id !== s.selected_token.token_id) fail(`step ${s.step}: selection mismatch`);
    if (sampling) {
      const inside = c.filter((x) => x.inside_top_p);
      const sum = inside.reduce((a, x) => a + (x.sampling_probability ?? 0), 0);
      if (inside.length !== s.nucleus_size || Math.abs(sum - 1) > TOL) fail(`step ${s.step}: nucleus not fully exported`);
      if (!sel[0].inside_top_p) fail(`step ${s.step}: selected token outside nucleus`);
    } else if (sel[0].rank !== 1) {
      fail(`step ${s.step}: greedy must select rank 1`);
    }
  }
  return t;
}

export const trace: InferenceTrace = loadInferenceTrace(sampledTraceJson);

if (trace.metadata.generation_mode !== "sampling") {
  // The main video explains temperature and top-p, which only exist in sampling mode.
  fail("the NextTokenDDR video needs a sampling trace in data/generated/inference_trace.json");
}

export const meta = trace.metadata;
export const firstStep: Step = trace.steps[0];

/** Top-ranked candidates for display. Always includes the selected token. */
export function displayCandidates(step: Step, k: number): Candidate[] {
  const selRank = step.selected_token.rank;
  return step.top_candidates.slice(0, Math.max(k, selRank));
}

/** Tokens of the DDR text itself (excludes any chat-template prefix). */
export const ddrTokens = trace.input_tokens.filter((t) => t.in_display_context);

/** Oil & Gas facts stated in the DDR text (NOT model parameters). */
export const ddrFacts = trace.source_context.facts;
