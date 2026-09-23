import { ddrFacts, trace } from "../data/loadInferenceTrace";
import { C, FONT_MONO, FONT_UI } from "../utils/theme";
import { Card } from "./Card";

/** Text each generated step appended, taken from the trace's own context strings. */
export const appendedTexts: string[] = trace.steps.map((s) => s.context_after.slice(s.context_before.length));

const [factKey, factValue] = Object.entries(ddrFacts)[0] ?? [null, null];

/** The DDR text as the viewer reads it, growing as generated tokens are appended. */
export const ContextDisplay: React.FC<{
  appended: number; // how many generated tokens to show
  flash?: number; // 0..1 highlight on the newest token
  fontSize?: number;
  tailHighlight?: number; // 0..1 highlight on the last words of the original context (scene 1)
  showFact?: boolean;
  cursor?: boolean;
  typed?: number; // typewriter: characters of the original context revealed (layout stays fixed)
}> = ({ appended, flash = 0, fontSize = 32, tailHighlight = 0, showFact = true, cursor = false, typed = Infinity }) => {
  const base = trace.display_context;
  const cut = base.lastIndexOf(". ") + 2; // highlight the unfinished last sentence
  const hidden = (from: number, to: number) => {
    const k = Math.max(from, Math.min(to, typed));
    return (
      <>
        {base.slice(from, k)}
        <span style={{ color: "transparent" }}>{base.slice(k, to)}</span>
      </>
    );
  };
  return (
    <Card style={{ padding: "22px 30px 26px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, gap: 16 }}>
        <div style={{ fontFamily: FONT_MONO, fontSize: 19, letterSpacing: 1.5, color: C.muted, fontWeight: 700 }}>
          DAILY DRILLING REPORT
          <div style={{ color: C.amber }}>SYNTHETIC EXAMPLE</div>
        </div>
        {showFact && factKey && (
          <div style={{ fontFamily: FONT_MONO, textAlign: "right", whiteSpace: "nowrap" }}>
            <div style={{ fontSize: 15, color: C.dim, letterSpacing: 1 }}>FROM REPORT TEXT · NOT A MODEL SETTING</div>
            <div style={{ fontSize: 22, color: C.teal, fontWeight: 700 }}>
              {factKey.replace(/_/g, " ").toUpperCase()} {factValue}
            </div>
          </div>
        )}
      </div>
      <div style={{ fontFamily: FONT_UI, fontSize, lineHeight: 1.42, color: C.text, whiteSpace: "pre-wrap", fontWeight: 500 }}>
        <span>{hidden(0, cut)}</span>
        <span
          style={{
            background: `rgba(34,211,238,${0.18 * tailHighlight})`,
            boxShadow: tailHighlight ? `0 0 0 4px rgba(34,211,238,${0.18 * tailHighlight})` : undefined,
            borderRadius: 6,
            color: tailHighlight ? C.cyan : C.text,
          }}
        >
          {hidden(cut, base.length)}
        </span>
        {appendedTexts.slice(0, appended).map((t, i) => {
          const newest = i === appended - 1;
          return (
            <span
              key={i}
              style={{
                color: C.magenta,
                fontWeight: 700,
                background: `rgba(232,121,249,${newest ? 0.12 + 0.3 * flash : 0.12})`,
                borderRadius: 6,
                boxShadow: `inset 0 -3px 0 ${C.magenta}88`,
              }}
            >
              {t}
            </span>
          );
        })}
        {cursor && typed >= base.length && <span style={{ color: C.cyan, fontWeight: 300 }}>▍</span>}
      </div>
    </Card>
  );
};
