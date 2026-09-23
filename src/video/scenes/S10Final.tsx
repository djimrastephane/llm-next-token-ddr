import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Card } from "../components/Card";
import { Tag } from "../components/Tag";
import { meta, trace } from "../data/loadInferenceTrace";
import { prog, rise, sceneFade } from "../utils/anim";
import { C, FONT_MONO, FONT_UI, PAD } from "../utils/theme";
import { sceneFrames } from "../utils/timeline";

const STEPS = [
  "Read the current context",
  "Calculate next-token scores",
  "Convert scores into probabilities",
  "Apply generation rules",
  "Select one token",
  "Append it to the context",
  "Repeat",
];
const STOP_NOTE: Record<string, string> = {
  sentence_end: "The model ended the sentence itself",
  eos: "The model emitted end-of-text",
  max_steps: "Generation was capped",
};

/** Numbers in the generated text (e.g. a pressure) are model output, not engineering results. */
export const GENERATED_VALUES_NOTE =
  "NUMBERS IN THE GENERATED TEXT ARE MODEL OUTPUT, NOT A HYDRAULICS CALCULATION OR MEASUREMENT";

/** Shown only when the generated text actually contains a number. */
export const HAS_NUMBERS = /\d/.test(trace.generated_text);

const REAL = ["REAL LOCAL MODEL", "REAL TOKENS", "REAL PROBABILITIES", "REAL SELECTION"];

export const S10Final: React.FC = () => {
  const f = useCurrentFrame();
  return (
    <AbsoluteFill style={{ opacity: sceneFade(f, sceneFrames("final")), padding: `0 ${PAD}px`, fontFamily: FONT_UI, justifyContent: "center" }}>
      <div style={{ ...rise(f, 0), fontSize: 84, fontWeight: 800, color: C.text, lineHeight: 1.02, letterSpacing: -1.5 }}>
        LLMs GENERATE TEXT
        <br />
        <span style={{ background: `linear-gradient(90deg, ${C.cyan}, ${C.purple})`, WebkitBackgroundClip: "text", color: "transparent" }}>
          ONE TOKEN AT A TIME
        </span>
      </div>
      <div style={{ marginTop: 44 }}>
        {STEPS.map((s, i) => (
          <div key={s} style={{ ...rise(f, 28 + i * 11, 22, 18), display: "flex", alignItems: "center", gap: 24, height: 78 }}>
            <span style={{ fontFamily: FONT_MONO, fontSize: 26, fontWeight: 800, color: C.bg, background: i === 6 ? C.magenta : C.cyan, width: 54, height: 54, borderRadius: 27, display: "flex", alignItems: "center", justifyContent: "center" }}>
              {i + 1}
            </span>
            <span style={{ fontSize: 42, color: C.text, fontWeight: 600 }}>{s}</span>
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginTop: 40 }}>
        {REAL.map((r, i) => (
          <div
            key={r}
            style={{
              ...rise(f, 125 + i * 10),
              fontFamily: FONT_MONO,
              fontSize: 30,
              fontWeight: 800,
              color: C.teal,
              border: `1.5px solid ${C.teal}66`,
              background: "rgba(45,212,191,0.08)",
              borderRadius: 16,
              padding: "22px 18px",
              textAlign: "center",
            }}
          >
            ✓ {r}
          </div>
        ))}
      </div>
      <Card style={{ ...rise(f, 175), marginTop: 30, padding: "22px 28px" }}>
        <div style={{ fontFamily: FONT_MONO, fontSize: 20, color: C.dim, letterSpacing: 1.5, fontWeight: 700 }}>
          {meta.model} · {meta.device} · T {meta.temperature?.toFixed(2)} · top-p {meta.top_p?.toFixed(2)} · seed {meta.seed}
        </div>
        <div style={{ fontSize: 36, color: C.text, marginTop: 12, lineHeight: 1.4 }}>
          …{trace.display_context.slice(trace.display_context.lastIndexOf(". ") + 1)}
          <span style={{ color: C.magenta, fontWeight: 700 }}>{trace.generated_text}</span>
        </div>
        <div style={{ fontSize: 22, color: C.dim, marginTop: 10, opacity: prog(f, 200, 20) }}>
          Synthetic DDR example. {STOP_NOTE[meta.stop_reason] ?? "Generation stopped"} after {meta.steps_generated} tokens.
        </div>
        {HAS_NUMBERS && (
          <Tag style={{ marginTop: 16, fontSize: 20, whiteSpace: "normal", lineHeight: 1.4, ...rise(f, 200) }}>
            {GENERATED_VALUES_NOTE}
          </Tag>
        )}
      </Card>
    </AbsoluteFill>
  );
};
