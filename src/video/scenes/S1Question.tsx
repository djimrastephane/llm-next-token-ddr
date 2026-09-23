import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { ContextDisplay } from "../components/ContextDisplay";
import { Tag } from "../components/Tag";
import { meta, trace } from "../data/loadInferenceTrace";
import { prog, rise, sceneFade } from "../utils/anim";
import { C, FONT_MONO, FONT_UI, PAD } from "../utils/theme";
import { sceneFrames } from "../utils/timeline";

export const S1Question: React.FC = () => {
  const f = useCurrentFrame();
  const typed = Math.floor(interpolate(f, [50, 125], [0, trace.display_context.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
  const q = prog(f, 175, 30);
  return (
    <AbsoluteFill style={{ opacity: sceneFade(f, sceneFrames("question")), padding: `300px ${PAD}px`, fontFamily: FONT_UI }}>
      <div style={{ ...rise(f, 0), fontFamily: FONT_MONO, fontSize: 26, color: C.cyan, letterSpacing: 3, fontWeight: 700 }}>
        REAL LOCAL INFERENCE · {meta.model_display_name.toUpperCase()}
      </div>
      <div style={{ ...rise(f, 6), fontSize: 100, fontWeight: 800, color: C.text, lineHeight: 1.0, letterSpacing: -2, marginTop: 22 }}>
        HOW LLMs PREDICT THE{" "}
        <span style={{ background: `linear-gradient(90deg, ${C.cyan}, ${C.purple})`, WebkitBackgroundClip: "text", color: "transparent" }}>
          NEXT TOKEN
        </span>
      </div>
      <div style={{ ...rise(f, 14), fontSize: 38, color: C.muted, marginTop: 28, lineHeight: 1.3, fontWeight: 500 }}>
        A real local LLM completing a Daily Drilling Report
      </div>
      <div style={{ ...rise(f, 36), marginTop: 70, position: "relative" }}>
        <ContextDisplay appended={0} fontSize={42} tailHighlight={prog(f, 135, 25)} showFact={false} typed={typed} cursor={Math.floor(f / 20) % 2 === 0} />
      </div>
      <div style={{ marginTop: 90, textAlign: "center", opacity: q, transform: `scale(${0.9 + 0.1 * q})` }}>
        <div style={{ fontSize: 96, fontWeight: 800, color: C.text, letterSpacing: -1 }}>
          WHAT COMES <span style={{ color: C.cyan, textShadow: `0 0 40px ${C.cyan}88` }}>NEXT?</span>
        </div>
        <Tag color={C.amber} style={{ marginTop: 30 }}>
          THE MODEL DECIDES · NOTHING BELOW IS HAND-WRITTEN
        </Tag>
      </div>
    </AbsoluteFill>
  );
};
