import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Card } from "../components/Card";
import type { BarSpec } from "../components/ProbabilityBar";
import { ProbabilityChart } from "../components/ProbabilityChart";
import { SceneTitle } from "../components/SceneTitle";
import { displayCandidates, firstStep } from "../data/loadInferenceTrace";
import { prog, rise, sceneFade } from "../utils/anim";
import { axisMax, int, pct } from "../utils/format";
import { C, CONTENT_W, FONT_MONO, FONT_UI } from "../utils/theme";
import { sceneFrames } from "../utils/timeline";
import { STAGE_LEFT, STAGE_TOP } from "./layout";

const K = 8;

export const S4Probabilities: React.FC = () => {
  const f = useCurrentFrame();
  const cands = displayCandidates(firstStep, K);
  const bars: BarSpec[] = cands.map((c) => ({ candidate: c, value: c.model_probability, tone: "cyan", label: pct(c.model_probability) }));
  const axis = axisMax(cands[0].model_probability);
  const shownMass = cands.reduce((a, c) => a + c.model_probability, 0);
  return (
    <AbsoluteFill style={{ opacity: sceneFade(f, sceneFrames("probabilities")), fontFamily: FONT_UI }}>
      <div style={{ position: "absolute", top: STAGE_TOP, left: STAGE_LEFT, width: CONTENT_W }}>
        <SceneTitle
          kicker="03 · SOFTMAX → PROBABILITIES"
          title="Next-token probabilities"
          subtitle={<>The model's own distribution for step t = 1 <span style={{ color: C.dim }}>(temperature 1)</span></>}
        />
        <div style={{ marginTop: 40 }}>
          <ProbabilityChart bars={bars} width={CONTENT_W} barH={560} axis={axis} grow={(i) => prog(f, 30 + i * 7, 45)} />
        </div>
        <div style={{ ...rise(f, 120), marginTop: 18, fontFamily: FONT_MONO, fontSize: 22, color: C.dim, textAlign: "center" }}>
          Top {cands.length} of {int(firstStep.vocab_size)} · the other {int(firstStep.vocab_size - cands.length)} share {pct(1 - shownMass)}
        </div>
        <Card style={{ ...rise(f, 160), marginTop: 26, padding: "20px 28px", fontSize: 30, lineHeight: 1.4, color: C.muted }}>
          Softmax only converts scores into probabilities. <b style={{ color: C.text }}>It does not choose the token.</b>{" "}
          The tallest bar is not automatically the one that gets picked.
        </Card>
      </div>
    </AbsoluteFill>
  );
};
