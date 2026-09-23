import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Card } from "../components/Card";
import { ModelBlock } from "../components/ModelBlock";
import { SceneTitle } from "../components/SceneTitle";
import { TokenText } from "../components/TokenText";
import { firstStep, trace } from "../data/loadInferenceTrace";
import { prog, rise, sceneFade } from "../utils/anim";
import { int, num } from "../utils/format";
import { C, CONTENT_W, FONT_MONO, FONT_UI } from "../utils/theme";
import { sceneFrames } from "../utils/timeline";
import { STAGE_LEFT, STAGE_TOP } from "./layout";

const SHOW = 5;

export const S3ForwardPass: React.FC = () => {
  const f = useCurrentFrame();
  const ids = trace.input_tokens.slice(-8);
  const pulse = interpolate(f, [30, 130], [0, 3], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }) % 1;
  const rest = firstStep.vocab_size - SHOW;
  return (
    <AbsoluteFill style={{ opacity: sceneFade(f, sceneFrames("forward")), fontFamily: FONT_UI }}>
      <div style={{ position: "absolute", top: STAGE_TOP, left: STAGE_LEFT, width: CONTENT_W }}>
        <SceneTitle kicker="02 · FORWARD PASS" title="The model scores every token" subtitle="The whole context goes in. One score per vocabulary entry comes out." />
        <div style={{ position: "relative", height: 120, marginTop: 20 }}>
          {ids.map((t, i) => {
            const p = prog(f, 18 + i * 6, 40);
            return (
              <div
                key={t.position}
                style={{
                  position: "absolute",
                  left: 20 + i * 118,
                  top: interpolate(p, [0, 1], [0, 90]),
                  opacity: p < 0.85 ? Math.min(1, p * 4) : (1 - p) / 0.15,
                  fontFamily: FONT_MONO,
                  fontSize: 26,
                  color: C.cyan,
                  border: `1.5px solid ${C.cyan}55`,
                  borderRadius: 12,
                  padding: "6px 10px",
                  background: "rgba(34,211,238,0.08)",
                }}
              >
                {t.token_id}
              </div>
            );
          })}
        </div>
        <div style={rise(f, 10)}>
          <ModelBlock pulse={f > 30 && f < 150 ? pulse : 0} width={CONTENT_W} />
        </div>
        <div style={{ ...rise(f, 20), textAlign: "center", fontSize: 40, color: C.dim, margin: "12px 0" }}>↓</div>
        <Card style={{ ...rise(f, 120), padding: "22px 30px" }} glow={C.purple}>
          <div style={{ fontFamily: FONT_MONO, fontSize: 24, color: C.purple, fontWeight: 700, letterSpacing: 2 }}>VOCABULARY SCORES (LOGITS)</div>
          <div style={{ fontSize: 28, color: C.muted, marginTop: 6 }}>
            <b style={{ color: C.text }}>{int(firstStep.vocab_size)}</b> raw scores, one per vocabulary entry. Highest shown:
          </div>
          <div style={{ marginTop: 12 }}>
            {firstStep.top_candidates.slice(0, SHOW).map((c, i) => (
              <div key={c.token_id} style={{ ...rise(f, 140 + i * 9, 18, 14), display: "flex", justifyContent: "space-between", alignItems: "center", height: 50, borderBottom: `1px solid ${C.border}` }}>
                <TokenText text={c.display_token} maxWidth={560} size={30} />
                <span style={{ fontFamily: FONT_MONO, fontSize: 30, color: C.text, fontWeight: 700 }}>{num(c.raw_logit)}</span>
              </div>
            ))}
            <div style={{ ...rise(f, 190), fontFamily: FONT_MONO, fontSize: 24, color: C.dim, marginTop: 12 }}>
              … and {int(rest)} more scores
            </div>
          </div>
        </Card>
      </div>
    </AbsoluteFill>
  );
};
