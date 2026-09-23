import { meta } from "../data/loadInferenceTrace";
import { C, FONT_MONO, FONT_UI } from "../utils/theme";
import { Tag } from "./Tag";

/** Conceptual box for the forward pass. Not a picture of real neural activity. */
export const ModelBlock: React.FC<{ pulse: number; width: number }> = ({ pulse, width }) => (
  <div
    style={{
      width,
      padding: "34px 30px",
      borderRadius: 32,
      background: "linear-gradient(160deg, rgba(34,211,238,0.10), rgba(167,139,250,0.14))",
      border: `2px solid ${C.cyan}${pulse > 0.5 ? "cc" : "66"}`,
      boxShadow: `0 0 ${20 + 50 * pulse}px rgba(34,211,238,${0.15 + 0.35 * pulse})`,
      textAlign: "center",
      fontFamily: FONT_UI,
    }}
  >
    <div style={{ fontSize: 58, fontWeight: 800, color: C.text, letterSpacing: 1 }}>LOCAL LLM</div>
    <div style={{ fontFamily: FONT_MONO, fontSize: 32, color: C.cyan, marginTop: 8, fontWeight: 700 }}>{meta.model_display_name}</div>
    <div style={{ fontFamily: FONT_MONO, fontSize: 22, color: C.muted, marginTop: 10 }}>
      running on {meta.device} · {meta.dtype} · no cloud API
    </div>
    <div style={{ display: "flex", justifyContent: "center", gap: 10, marginTop: 20 }}>
      {Array.from({ length: 9 }).map((_, i) => (
        <div
          key={i}
          style={{
            width: 64,
            height: 12,
            borderRadius: 6,
            background: C.purple,
            opacity: 0.25 + 0.6 * Math.max(0, Math.sin(pulse * Math.PI * 2 + i * 0.7)),
          }}
        />
      ))}
    </div>
    <Tag style={{ marginTop: 22 }}>CONCEPTUAL VIEW · NOT LITERAL NEURAL ACTIVITY</Tag>
  </div>
);
