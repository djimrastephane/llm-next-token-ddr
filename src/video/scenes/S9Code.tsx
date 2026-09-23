import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Card } from "../components/Card";
import { CodeEditor, type CodeLine } from "../components/CodeEditor";
import { SceneTitle } from "../components/SceneTitle";
import { Tag } from "../components/Tag";
import { meta } from "../data/loadInferenceTrace";
import { rise, sceneFade } from "../utils/anim";
import { C, CONTENT_W, FONT_MONO, FONT_UI } from "../utils/theme";
import { sceneFrames } from "../utils/timeline";
import { STAGE_LEFT, STAGE_TOP } from "./layout";

// Field names of every candidate in data/generated/inference_trace.json (schema 1.0).
const RECORDED = [
  "token_id",
  "decoded_token",
  "raw_logit",
  "model_probability",
  "temperature_probability",
  "cumulative_probability",
  "inside_top_p",
  "sampling_probability",
  "selected",
];

const LINES: CodeLine[] = [
  { text: "def generate(context):", indent: 0 },
  { text: "while not finished:", indent: 1 },
  { text: "logits = model(context)", indent: 2 },
  { text: "probs = softmax(logits / temperature)", indent: 2 },
  { text: `nucleus = top_p(probs, p=${meta.top_p?.toFixed(2)})`, indent: 2 },
  { text: "next_token = sample(nucleus)", indent: 2 },
  { text: "context.append(next_token)", indent: 2 },
];

export const S9Code: React.FC = () => {
  const f = useCurrentFrame();
  const total = LINES.reduce((a, l) => a + l.text.length, 0);
  const typed = interpolate(f, [16, 130], [0, total], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const active = f < 135 ? -1 : 2 + (Math.floor((f - 135) / 22) % 5);
  return (
    <AbsoluteFill style={{ opacity: sceneFade(f, sceneFrames("code")), fontFamily: FONT_UI }}>
      <div style={{ position: "absolute", top: STAGE_TOP, left: STAGE_LEFT, width: CONTENT_W }}>
        <SceneTitle kicker="08 · THE LOOP IN CODE" title="Simplified generation logic" />
        <div style={{ ...rise(f, 8), marginTop: 36 }}>
          <CodeEditor file="llm_generate.py" lines={LINES} typed={typed} active={active} width={CONTENT_W} />
        </div>
        <div style={{ ...rise(f, 20), marginTop: 28, display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 14 }}>
          <Tag>SIMPLIFIED · PSEUDOCODE</Tag>
          <div style={{ fontSize: 27, color: C.muted, lineHeight: 1.4 }}>
            Not the literal Hugging Face implementation. The real capture script does the same steps on the full vocabulary.
          </div>
        </div>
        <Card style={{ ...rise(f, 60), marginTop: 34, padding: "22px 26px" }}>
          <div style={{ fontFamily: FONT_MONO, fontSize: 21, color: C.cyan, fontWeight: 800, letterSpacing: 1.5 }}>
            RECORDED FOR EVERY CANDIDATE, EVERY STEP
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 16 }}>
            {RECORDED.map((k, i) => (
              <span key={k} style={{ ...rise(f, 70 + i * 5, 16, 10), fontFamily: FONT_MONO, fontSize: 23, color: C.text, background: C.surfaceHi, border: `1px solid ${C.border}`, borderRadius: 10, padding: "6px 12px" }}>
                {k}
              </span>
            ))}
          </div>
          <div style={{ fontFamily: FONT_MONO, fontSize: 21, color: C.dim, marginTop: 16 }}>→ data/generated/inference_trace.json</div>
        </Card>
      </div>
    </AbsoluteFill>
  );
};
