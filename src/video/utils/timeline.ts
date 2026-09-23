import { trace } from "../data/loadInferenceTrace";

// Scene lengths in frames at 60 fps. The loop scene grows with the number of captured steps.
export const LOOP_FIRST = 170; // step 1: "context + token -> new context"
export const LOOP_DETAILED = 115; // design length of one later step; its animation is timed against this
const DETAILED_STEPS = 3; // t = 2..4 at full pace, the rest in fast-forward
const LOOP_FAST = 62;
const APPEND_AT = 88; // design frame at which the selected token lands in the context

const loopSteps = trace.steps.length - 1;

/** Length of later step i (i = 0 is t = 2). */
export const loopStepFrames = (i: number): number => (i < DETAILED_STEPS ? LOOP_DETAILED : LOOP_FAST);

/** Start of later step i, relative to the loop scene. */
export const loopStepStart = (i: number): number => {
  let f = LOOP_FIRST;
  for (let k = 0; k < i; k++) f += loopStepFrames(k);
  return f;
};

/** Map a step's local frame onto the LOOP_DETAILED design clock (fast steps play the same animation quicker). */
export const designFrame = (local: number, i: number): number => (local * LOOP_DETAILED) / loopStepFrames(i);

export const SCENES = [
  { id: "question", frames: 260 },
  { id: "tokenization", frames: 300 },
  { id: "forward", frames: 230 },
  { id: "probabilities", frames: 320 },
  { id: "temperature", frames: 340 },
  { id: "topP", frames: 380 },
  { id: "selection", frames: 330 },
  { id: "loop", frames: loopStepStart(loopSteps) },
  { id: "code", frames: 200 },
  { id: "final", frames: 380 },
] as const;

export type SceneId = (typeof SCENES)[number]["id"];

export const sceneStart = (id: SceneId): number => {
  let f = 0;
  for (const s of SCENES) {
    if (s.id === id) return f;
    f += s.frames;
  }
  throw new Error(id);
};

export const sceneFrames = (id: SceneId): number => SCENES.find((s) => s.id === id)!.frames;

export const TOTAL_FRAMES = SCENES.reduce((a, s) => a + s.frames, 0);

/** Which generation step (t) the HUD should show at a global frame. */
export function hudStep(frame: number): number {
  const local = frame - sceneStart("loop");
  if (local < LOOP_FIRST) return 1;
  if (frame >= sceneStart("code")) return trace.steps.length;
  let i = 0;
  while (i < loopSteps - 1 && local >= loopStepStart(i + 1)) i++;
  return i + 2;
}

/** Number of generated tokens already appended to the context at a global frame. */
export function appendedCount(frame: number): number {
  const loop0 = sceneStart("loop");
  if (frame < loop0) return 0;
  const appendAt = (i: number) =>
    i === 0 ? loop0 + 95 : loop0 + loopStepStart(i - 1) + (APPEND_AT * loopStepFrames(i - 1)) / LOOP_DETAILED;
  let n = 0;
  for (let i = 0; i < trace.steps.length; i++) if (frame >= appendAt(i)) n = i + 1;
  return n;
}
