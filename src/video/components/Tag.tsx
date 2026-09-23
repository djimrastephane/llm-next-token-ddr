import { C, FONT_MONO } from "../utils/theme";

/** Small uppercase pill, e.g. SIMPLIFIED or CONCEPTUAL VIEW. */
export const Tag: React.FC<{ children: React.ReactNode; color?: string; style?: React.CSSProperties }> = ({
  children,
  color = C.amber,
  style,
}) => (
  <span
    style={{
      display: "inline-block",
      fontFamily: FONT_MONO,
      fontSize: 22,
      fontWeight: 700,
      letterSpacing: 1.5,
      color,
      border: `1.5px solid ${color}66`,
      background: `${color}14`,
      borderRadius: 999,
      padding: "6px 16px",
      ...style,
    }}
  >
    {children}
  </span>
);
