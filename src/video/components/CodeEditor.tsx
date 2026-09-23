import { C, FONT_MONO } from "../utils/theme";

export type CodeLine = { text: string; indent: number };

function highlight(line: string) {
  // Tiny tokenizer for the pseudocode: keywords purple, calls cyan, numbers amber, comments grey.
  const out: React.ReactNode[] = [];
  const re = /(#.*$)|(\b(?:def|while|not)\b)|(\b[a-z_]+(?=\())|(\b\d+(?:\.\d+)?\b)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(line))) {
    if (m.index > last) out.push(line.slice(last, m.index));
    const color = m[1] ? C.dim : m[2] ? C.purple : m[3] ? C.cyan : C.amber;
    out.push(
      <span key={k++} style={{ color }}>
        {m[0]}
      </span>,
    );
    last = m.index + m[0].length;
  }
  out.push(line.slice(last));
  return out;
}

/** macOS-style editor window. `typed` = number of characters revealed; `active` = highlighted line. */
export const CodeEditor: React.FC<{ file: string; lines: CodeLine[]; typed: number; active: number; width: number }> = ({
  file,
  lines,
  typed,
  active,
  width,
}) => {
  let budget = typed;
  return (
    <div style={{ width, borderRadius: 22, overflow: "hidden", border: `1.5px solid ${C.borderHi}`, background: "#0F1523", boxShadow: "0 30px 80px rgba(0,0,0,0.5)" }}>
      <div style={{ height: 62, display: "flex", alignItems: "center", gap: 12, padding: "0 22px", background: "#161E2E", borderBottom: `1.5px solid ${C.border}` }}>
        {[C.red, C.yellow, C.green].map((c) => (
          <div key={c} style={{ width: 20, height: 20, borderRadius: 10, background: c }} />
        ))}
        <div style={{ marginLeft: 22, padding: "8px 20px", borderRadius: 10, background: "#0F1523", fontFamily: FONT_MONO, fontSize: 22, color: C.text }}>
          {file}
        </div>
      </div>
      <div style={{ padding: "24px 0 28px", fontFamily: FONT_MONO, fontSize: 29, lineHeight: 1.75 }}>
        {lines.map((l, i) => {
          const visible = l.text.slice(0, Math.max(0, budget));
          budget -= l.text.length;
          return (
            <div key={i} style={{ display: "flex", background: i === active ? "rgba(34,211,238,0.10)" : undefined, borderLeft: `4px solid ${i === active ? C.cyan : "transparent"}` }}>
              <span style={{ width: 64, textAlign: "right", color: C.dim, paddingRight: 20 }}>{i + 1}</span>
              <span style={{ color: C.text, paddingLeft: l.indent * 30, whiteSpace: "pre" }}>{highlight(visible)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
