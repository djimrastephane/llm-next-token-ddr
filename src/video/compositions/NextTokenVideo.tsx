import { AbsoluteFill, Sequence, useCurrentFrame } from "remotion";
import { Background } from "../components/Background";
import { ContextDisplay } from "../components/ContextDisplay";
import { HUD } from "../components/HUD";
import { S10Final } from "../scenes/S10Final";
import { S1Question } from "../scenes/S1Question";
import { S2Tokenization } from "../scenes/S2Tokenization";
import { S3ForwardPass } from "../scenes/S3ForwardPass";
import { S4Probabilities } from "../scenes/S4Probabilities";
import { S5Temperature } from "../scenes/S5Temperature";
import { S6TopP } from "../scenes/S6TopP";
import { S7Selection } from "../scenes/S7Selection";
import { S8Loop } from "../scenes/S8Loop";
import { S8bCompare } from "../scenes/S8bCompare";
import { S9Code } from "../scenes/S9Code";
import { prog } from "../utils/anim";
import { PAD } from "../utils/theme";
import { SCENES, type SceneId, appendedCount, hudStep, sceneStart } from "../utils/timeline";

const SCENE_COMPONENTS: Record<SceneId, React.FC> = {
  question: S1Question,
  tokenization: S2Tokenization,
  forward: S3ForwardPass,
  probabilities: S4Probabilities,
  temperature: S5Temperature,
  topP: S6TopP,
  selection: S7Selection,
  loop: S8Loop,
  compare: S8bCompare,
  code: S9Code,
  final: S10Final,
};

/** Frames since `fn(frame)` last changed, capped at `max`. Drives short highlight flashes. */
function sinceChange(frame: number, fn: (f: number) => number, max = 45): number {
  const now = fn(frame);
  for (let k = 1; k <= max; k++) if (fn(frame - k) !== now) return k;
  return max;
}

export const NextTokenVideo: React.FC = () => {
  const frame = useCurrentFrame();
  // HUD + DDR card: shown from tokenization to the end of the code scene, except during the comparison
  // scene, which shows both runs with their own context.
  const visible = (from: number, to: number) => Math.min(prog(frame, from, 16), 1 - prog(frame, to - 16, 16));
  const overlay = Math.max(
    visible(sceneStart("tokenization"), sceneStart("compare")),
    visible(sceneStart("code"), sceneStart("final")),
  );
  const flash = 1 - sinceChange(frame, appendedCount) / 45;
  const stepFlash = 1 - sinceChange(frame, hudStep) / 45;
  let from = 0;
  return (
    <AbsoluteFill>
      <Background />
      {SCENES.map((s) => {
        const Scene = SCENE_COMPONENTS[s.id];
        const seq = (
          <Sequence key={s.id} from={from} durationInFrames={s.frames} name={s.id}>
            <Scene />
          </Sequence>
        );
        from += s.frames;
        return seq;
      })}
      {overlay > 0 && (
        <>
          <div style={{ position: "absolute", top: 44, left: PAD, opacity: overlay }}>
            <HUD step={hudStep(frame)} stepFlash={stepFlash} />
          </div>
          <div style={{ position: "absolute", bottom: 44, left: PAD, right: PAD, opacity: overlay }}>
            <ContextDisplay appended={appendedCount(frame)} flash={flash} />
          </div>
        </>
      )}
    </AbsoluteFill>
  );
};
