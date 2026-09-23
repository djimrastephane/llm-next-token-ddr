import { trace } from "../data/loadInferenceTrace";

// Scene lengths in frames at 60 fps. The loop scene grows with the number of captured steps.
export const LOOP_FIRST = 170;
export const LOOP_EACH = 115;

const loopSteps = trace.steps.length - 1;

export const SCENES = [
  { id: "question", frames: 280 },
  { id: "tokenization", frames: 320 },
  { id: "forward", frames: 250 },
  { id: "probabilities", frames: 340 },
  { id: "temperature", frames: 360 },
  { id: "topP", frames: 400 },
  { id: "selection", frames: 350 },
  { id: "loop", frames: LOOP_FIRST + loopSteps * LOOP_EACH },
  { id: "code", frames: 230 },
  { id: "final", frames: 400 },
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
  const loop0 = sceneStart("loop");
  if (frame < loop0 + LOOP_FIRST) return 1;
  if (frame >= sceneStart("code")) return trace.steps.length;
  return Math.min(trace.steps.length, 2 + Math.floor((frame - loop0 - LOOP_FIRST) / LOOP_EACH));
}

/** Number of generated tokens already appended to the context at a global frame. */
export function appendedCount(frame: number): number {
  const loop0 = sceneStart("loop");
  if (frame < loop0) return 0;
  const appendAt = (i: number) => (i === 0 ? loop0 + 95 : loop0 + LOOP_FIRST + (i - 1) * LOOP_EACH + 88);
  let n = 0;
  for (let i = 0; i < trace.steps.length; i++) if (frame >= appendAt(i)) n = i + 1;
  return n;
}
