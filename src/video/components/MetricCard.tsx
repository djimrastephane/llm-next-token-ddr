import { C, FONT_MONO } from "../utils/theme";

export const MetricCard: React.FC<{ label: string; value: React.ReactNode; accent?: string; flex?: number; highlight?: number }> = ({
  label,
  value,
  accent = C.text,
  flex = 1,
  highlight = 0,
}) => (
  <div
    style={{
      flex,
      minWidth: 0,
      background: C.surface,
      border: `1.5px solid ${highlight > 0 ? accent : C.border}`,
      boxShadow: highlight > 0 ? `0 0 ${24 * highlight}px ${accent}55` : undefined,
      borderRadius: 18,
      padding: "12px 16px",
      fontFamily: FONT_MONO,
    }}
  >
    <div style={{ fontSize: 18, color: C.dim, letterSpacing: 1.5, fontWeight: 700, whiteSpace: "nowrap" }}>{label}</div>
    <div style={{ fontSize: 30, color: accent, fontWeight: 700, marginTop: 4, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
      {value}
    </div>
  </div>
);
