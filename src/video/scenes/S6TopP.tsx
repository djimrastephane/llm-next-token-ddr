import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Card } from "../components/Card";
import type { BarSpec } from "../components/ProbabilityBar";
import { ProbabilityChart, slotCenter } from "../components/ProbabilityChart";
import { SceneTitle } from "../components/SceneTitle";
import { firstStep, meta } from "../data/loadInferenceTrace";
import { easeInOut, prog, rise, sceneFade } from "../utils/anim";
import { axisMax, pct } from "../utils/format";
import { C, CONTENT_W, FONT_MONO, FONT_UI } from "../utils/theme";
import { sceneFrames } from "../utils/timeline";
import { TopPNucleus } from "../components/TopPNucleus";
import { STAGE_LEFT, STAGE_TOP } from "./layout";

const REVEAL_START = 60;
const REVEAL_EACH = 18;
const RENORM = 265;

export const S6TopP: React.FC = () => {
  const f = useCurrentFrame();
  const P = meta.top_p!;
  const n = firstStep.nucleus_size!;
  const cands = firstStep.top_candidates.slice(0, Math.min(10, Math.max(8, n + 2)));
  const reveal = interpolate(f, [REVEAL_START, REVEAL_START + cands.length * REVEAL_EACH], [0, cands.length], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const renorm = prog(f, RENORM, 50, easeInOut);
  const bars: BarSpec[] = cands.map((c, i) => {
    const decided = reveal >= i + 1;
    const out = decided && !c.inside_top_p;
    const sp = c.sampling_probability ?? 0;
    return {
      candidate: c,
      value: c.inside_top_p ? interpolate(renorm, [0, 1], [c.temperature_probability, sp]) : c.temperature_probability,
      tone: out ? "excluded" : "cyan",
      label: c.inside_top_p && renorm > 0.5 ? pct(sp) : pct(c.temperature_probability),
      label2: c.inside_top_p && renorm > 0.5 ? `was ${pct(c.temperature_probability)}` : undefined,
      footer: out ? "EXCLUDED" : decided ? `Σ ${pct(c.cumulative_probability)}` : `rank ${c.rank}`,
      footerColor: out ? "#64748B" : decided ? C.cyan : undefined,
      opacity: out ? 0.75 : 1,
    };
  });
  const axis = axisMax(Math.max(...cands.map((c) => Math.max(c.temperature_probability, c.sampling_probability ?? 0))));
  const bracket = prog(f, REVEAL_START + cands.length * REVEAL_EACH + 10, 24);
  const shownEligible = cands.filter((c) => c.inside_top_p).length;
  const bracketW = slotCenter(shownEligible - 1, cands.length, CONTENT_W) + CONTENT_W / cands.length / 2;
  return (
    <AbsoluteFill style={{ opacity: sceneFade(f, sceneFrames("topP")), fontFamily: FONT_UI }}>
      <div style={{ position: "absolute", top: STAGE_TOP, left: STAGE_LEFT, width: CONTENT_W }}>
        <SceneTitle
          kicker="05 · TOP-P (NUCLEUS)"
          title={<>Top-p = <span style={{ color: C.magenta }}>{P.toFixed(2)}</span></>}
          subtitle={`Keep the smallest set of top tokens whose probabilities add up to at least ${pct(P)}.`}
        />
        <div style={{ ...rise(f, 30), marginTop: 30 }}>
          <TopPNucleus candidates={cands} topP={P} reveal={reveal} width={CONTENT_W} />
        </div>
        <div style={{ height: 60, position: "relative", opacity: bracket }}>
          <div style={{ position: "absolute", left: 0, width: bracketW, top: 38, height: 16, border: `3px solid ${C.cyan}`, borderBottom: "none", borderRadius: "10px 10px 0 0" }} />
          <div style={{ position: "absolute", left: 0, width: bracketW, top: 0, textAlign: "center", fontFamily: FONT_MONO, fontSize: 23, color: C.cyan, fontWeight: 800 }}>
            ELIGIBLE FOR SAMPLING · {n} TOKEN{n === 1 ? "" : "S"}
          </div>
        </div>
        <div style={rise(f, 30)}>
          <ProbabilityChart bars={bars} width={CONTENT_W} barH={360} axis={axis} grow={() => prog(f, 30, 30)} />
        </div>
        <Card style={{ ...rise(f, RENORM + 10), marginTop: 20, padding: "16px 24px", fontSize: 25, lineHeight: 1.4, color: C.muted }}>
          <b style={{ color: C.text }}>Sampling probability</b> = probability ÷ {pct(firstStep.nucleus_temperature_mass!)} (the eligible total).
          Excluded tokens get 0%. The model's own probabilities are unchanged.
        </Card>
      </div>
    </AbsoluteFill>
  );
};
