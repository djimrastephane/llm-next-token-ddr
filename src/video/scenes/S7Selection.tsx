import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Card } from "../components/Card";
import type { BarSpec } from "../components/ProbabilityBar";
import { ProbabilityChart, slotCenter } from "../components/ProbabilityChart";
import { SceneTitle } from "../components/SceneTitle";
import { SelectionClaw } from "../components/SelectionClaw";
import { Tag } from "../components/Tag";
import { TokenText } from "../components/TokenText";
import { firstStep, meta, trace, withSelected } from "../data/loadInferenceTrace";
import { prog, rise, sceneFade } from "../utils/anim";
import { clawPath, clawSlot } from "../utils/claw";
import { axisMax, pct } from "../utils/format";
import { C, CONTENT_W, FONT_MONO, FONT_UI } from "../utils/theme";
import { sceneFrames } from "../utils/timeline";
import { STAGE_LEFT, STAGE_TOP } from "./layout";

const LOCK = 190;

// A later step (if any) where sampling picked a token other than rank 1, straight from the trace.
const laterNonTop = trace.steps.find((s) => s.step > 1 && s.selected_token.rank > 1);

const Stat: React.FC<{ label: string; value: string; color?: string }> = ({ label, value, color = C.text }) => (
  <div style={{ flex: 1, textAlign: "center" }}>
    <div style={{ fontFamily: FONT_MONO, fontSize: 18, color: C.dim, letterSpacing: 1, fontWeight: 700 }}>{label}</div>
    <div style={{ fontFamily: FONT_MONO, fontSize: 38, color, fontWeight: 800, marginTop: 4 }}>{value}</div>
  </div>
);

export const S7Selection: React.FC = () => {
  const f = useCurrentFrame();
  const eligible = withSelected(firstStep.top_candidates.filter((c) => c.inside_top_p), 8);
  const sel = firstStep.selected_token;
  const selIdx = eligible.findIndex((c) => c.selected);
  const lock = prog(f, LOCK, 20);
  const bars: BarSpec[] = eligible.map((c) => ({
    candidate: c,
    value: c.sampling_probability!,
    tone: c.selected && lock > 0.3 ? "selected" : "cyan",
    label: pct(c.sampling_probability!),
  }));
  const axis = axisMax(Math.max(...bars.map((b) => b.value)));
  const slot = clawSlot(clawPath(eligible.length, selIdx), prog(f, 45, LOCK - 45));
  return (
    <AbsoluteFill style={{ opacity: sceneFade(f, sceneFrames("selection")), fontFamily: FONT_UI }}>
      <div style={{ position: "absolute", top: STAGE_TOP, left: STAGE_LEFT, width: CONTENT_W }}>
        <SceneTitle kicker="06 · SELECTION" title="One eligible token is drawn" subtitle="A random draw weighted by sampling probability." />
        <div style={{ position: "relative", paddingTop: 170, marginTop: 10 }}>
          <div style={{ position: "absolute", inset: 0, opacity: prog(f, 30, 16) }}>
            <SelectionClaw x={slotCenter(slot, eligible.length, CONTENT_W)} drop={60} grip={prog(f, LOCK - 8, 14)} glow={lock} />
          </div>
          <div style={rise(f, 14)}>
            <ProbabilityChart bars={bars} width={CONTENT_W} barH={330} axis={axis} grow={(i) => prog(f, 14 + i * 3, 26)} glow={(i) => (i === selIdx ? lock : 0)} />
          </div>
        </div>
        <div style={{ ...rise(f, 20), textAlign: "center", marginTop: 12 }}>
          <Tag style={{ fontSize: 19 }}>CONCEPTUAL VIEW · CLAW MOTION IS ILLUSTRATIVE · ONE SEEDED DRAW (SEED {meta.seed})</Tag>
        </div>
        <Card glow={C.magenta} style={{ ...rise(f, LOCK + 16), marginTop: 26, padding: "22px 26px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ fontFamily: FONT_MONO, fontSize: 24, color: C.magenta, fontWeight: 800, letterSpacing: 2 }}>SELECTED TOKEN</div>
            <div style={{ padding: "8px 22px", borderRadius: 16, border: `2px solid ${C.magenta}`, background: "rgba(232,121,249,0.12)" }}>
              <TokenText text={sel.display_token} maxWidth={440} size={54} weight={800} />
            </div>
          </div>
          <div style={{ display: "flex", marginTop: 20, gap: 10 }}>
            <Stat label="RANK" value={String(sel.rank)} />
            <Stat label="MODEL PROB." value={pct(sel.model_probability)} />
            <Stat label="SAMPLING PROB." value={pct(sel.sampling_probability!)} color={C.magenta} />
          </div>
          <div style={{ marginTop: 18, fontSize: 25, color: C.muted, lineHeight: 1.4, ...rise(f, LOCK + 40) }}>
            {sel.rank === 1
              ? `This draw landed on rank 1, but any eligible token could have been drawn.${
                  laterNonTop ? ` At step t = ${laterNonTop.step} the draw lands on rank ${laterNonTop.selected_token.rank}.` : ""
                }`
              : `Not the most likely token: the draw landed on rank ${sel.rank}. Sampling keeps less likely options possible.`}
          </div>
        </Card>
      </div>
    </AbsoluteFill>
  );
};
