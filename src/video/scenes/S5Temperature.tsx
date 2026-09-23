import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Card } from "../components/Card";
import type { BarSpec } from "../components/ProbabilityBar";
import { ProbabilityChart } from "../components/ProbabilityChart";
import { SceneTitle } from "../components/SceneTitle";
import { TokenText } from "../components/TokenText";
import { displayCandidates, firstStep, meta } from "../data/loadInferenceTrace";
import { easeInOut, prog, rise, sceneFade } from "../utils/anim";
import { axisMax, num, pct } from "../utils/format";
import { C, CONTENT_W, FONT_MONO, FONT_UI } from "../utils/theme";
import { sceneFrames } from "../utils/timeline";
import { STAGE_LEFT, STAGE_TOP } from "./layout";

const K = 8;

const Box: React.FC<{ label: string; value: React.ReactNode; color?: string; style: React.CSSProperties }> = ({ label, value, color = C.text, style }) => (
  <div style={{ ...style, flex: 1, textAlign: "center", background: C.surfaceHi, border: `1.5px solid ${C.border}`, borderRadius: 16, padding: "12px 6px" }}>
    <div style={{ fontFamily: FONT_MONO, fontSize: 17, color: C.dim, letterSpacing: 1, fontWeight: 700 }}>{label}</div>
    <div style={{ fontFamily: FONT_MONO, fontSize: 32, color, fontWeight: 700, marginTop: 4 }}>{value}</div>
  </div>
);

const Op: React.FC<{ children: React.ReactNode; style: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ ...style, fontFamily: FONT_MONO, fontSize: 22, color: C.purple, fontWeight: 700, textAlign: "center", width: 96 }}>{children}</div>
);

export const S5Temperature: React.FC = () => {
  const f = useCurrentFrame();
  const T = meta.temperature!;
  const cands = displayCandidates(firstStep, K);
  const top = cands[0];
  const morph = prog(f, 175, 60, easeInOut);
  const bars: BarSpec[] = cands.map((c) => ({
    candidate: c,
    value: interpolate(morph, [0, 1], [c.model_probability, c.temperature_probability]),
    ghost: morph > 0 ? c.model_probability : undefined,
    tone: "purple",
    label: pct(morph < 0.5 ? c.model_probability : c.temperature_probability),
  }));
  const axis = axisMax(Math.max(top.model_probability, top.temperature_probability));
  const effect = T < 1 ? "sharpens" : T > 1 ? "flattens" : "leaves unchanged";
  return (
    <AbsoluteFill style={{ opacity: sceneFade(f, sceneFrames("temperature")), fontFamily: FONT_UI }}>
      <div style={{ position: "absolute", top: STAGE_TOP, left: STAGE_LEFT, width: CONTENT_W }}>
        <SceneTitle
          kicker="04 · TEMPERATURE"
          title={<>Temperature = <span style={{ color: C.purple }}>{T.toFixed(2)}</span></>}
          subtitle="Changes how concentrated the distribution is."
        />
        <Card style={{ ...rise(f, 16), marginTop: 30, padding: "18px 20px" }}>
          <div style={{ fontFamily: FONT_MONO, fontSize: 20, color: C.muted, marginBottom: 12, display: "flex", alignItems: "center", gap: 12 }}>
            FOR TOKEN <TokenText text={top.display_token} maxWidth={300} size={24} color={C.text} /> (RANK 1)
          </div>
          <div style={{ display: "flex", alignItems: "center" }}>
            <Box label="RAW LOGIT" value={num(top.raw_logit)} style={rise(f, 26)} />
            <Op style={rise(f, 40)}>÷ {T.toFixed(2)}</Op>
            <Box label="SCALED" value={num(top.scaled_logit)} style={rise(f, 54)} />
            <Op style={rise(f, 68)}>SOFTMAX</Op>
            <Box label="PROBABILITY" value={pct(top.temperature_probability)} color={C.purple} style={rise(f, 82)} />
          </div>
        </Card>
        <div style={{ ...rise(f, 100), marginTop: 26 }}>
          <ProbabilityChart bars={bars} width={CONTENT_W} barH={430} axis={axis} grow={(i) => prog(f, 100 + i * 4, 30)} />
        </div>
        <div style={{ ...rise(f, 175), marginTop: 12, display: "flex", justifyContent: "center", gap: 36, fontFamily: FONT_MONO, fontSize: 21, color: C.muted }}>
          <span>
            <span style={{ display: "inline-block", width: 26, height: 16, border: `2.5px dashed ${C.muted}`, marginRight: 10 }} />
            temperature 1.00
          </span>
          <span>
            <span style={{ display: "inline-block", width: 26, height: 16, background: C.purple, marginRight: 10 }} />
            temperature {T.toFixed(2)}
          </span>
        </div>
        <Card style={{ ...rise(f, 250), marginTop: 20, padding: "18px 26px", fontSize: 28, lineHeight: 1.4, color: C.muted }}>
          T = {T.toFixed(2)} {effect} it: rank 1 goes from {pct(top.model_probability)} to{" "}
          <b style={{ color: C.text }}>{pct(top.temperature_probability)}</b>. Temperature adds no randomness by itself.
        </Card>
      </div>
    </AbsoluteFill>
  );
};
