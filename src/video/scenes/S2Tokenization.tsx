import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Card } from "../components/Card";
import { SceneTitle } from "../components/SceneTitle";
import { TokenizerView } from "../components/TokenizerView";
import { ddrTokens, trace } from "../data/loadInferenceTrace";
import { rise, sceneFade } from "../utils/anim";
import { C, CONTENT_W, FONT_MONO, FONT_UI } from "../utils/theme";
import { sceneFrames } from "../utils/timeline";
import { STAGE_LEFT, STAGE_TOP } from "./layout";

const Arrow: React.FC<{ style: React.CSSProperties }> = ({ style }) => (
  <div style={{ textAlign: "center", fontSize: 40, color: C.dim, lineHeight: 1, margin: "10px 0", ...style }}>↓</div>
);

export const S2Tokenization: React.FC = () => {
  const f = useCurrentFrame();
  const words = trace.display_context.trim().split(/\s+/).length;
  return (
    <AbsoluteFill style={{ opacity: sceneFade(f, sceneFrames("tokenization")), fontFamily: FONT_UI }}>
      <div style={{ position: "absolute", top: STAGE_TOP, left: STAGE_LEFT, width: CONTENT_W }}>
        <SceneTitle kicker="01 · TOKENIZATION" title="Text → tokens → token IDs" subtitle="LLMs process tokens, not words." />
        <div style={{ marginTop: 34 }}>
          <Card style={{ ...rise(f, 18), padding: "16px 26px", fontSize: 30, color: C.text, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
            <span style={{ fontFamily: FONT_MONO, fontSize: 20, color: C.dim, marginRight: 16 }}>TEXT</span>
            {trace.display_context}
          </Card>
          <Arrow style={rise(f, 30)} />
          <div style={{ ...rise(f, 36), display: "flex", justifyContent: "center" }}>
            <div style={{ fontFamily: FONT_MONO, fontSize: 28, fontWeight: 700, color: C.bg, background: C.cyan, borderRadius: 14, padding: "10px 30px" }}>
              TOKENIZER
            </div>
          </div>
          <Arrow style={rise(f, 46)} />
          <TokenizerView tokens={ddrTokens} frame={f} start={60} every={4} width={CONTENT_W} />
          <div style={{ ...rise(f, 70 + ddrTokens.length * 4), marginTop: 26, display: "flex", justifyContent: "space-between", fontFamily: FONT_MONO, fontSize: 24, color: C.muted }}>
            <span>
              <b style={{ color: C.text }}>{ddrTokens.length} tokens</b> for {words} words
            </span>
            <span>
              <span style={{ color: C.dim }}>␠</span> = space inside the token
            </span>
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
