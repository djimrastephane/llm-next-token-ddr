import { C, FONT_MONO } from "../utils/theme";
import { clampToken, fitMono } from "../utils/format";

const MARKERS = new Set(["␠", "↵", "⇥", "␍", "∅"]);

/**
 * Renders a trace `display_token`. Visible-whitespace markers are dimmed so the viewer can tell
 * "␠pressure" is a single token containing a leading space. Byte fragments (<0xNN>) are shown in amber.
 */
export const TokenText: React.FC<{ text: string; maxWidth: number; size: number; minSize?: number; color?: string; weight?: number }> = ({
  text,
  maxWidth,
  size,
  minSize = 16,
  color = C.text,
  weight = 600,
}) => {
  const onlyMarkers = Array.from(text).every((ch) => MARKERS.has(ch));
  const fontSize = fitMono(text, maxWidth, size, minSize) * (onlyMarkers ? 1.5 : 1);
  const shown = clampToken(text, maxWidth, minSize);
  const parts = shown.split(/(<0x[0-9A-F]{2}>|<U\+[0-9A-F]{4}>)/);
  return (
    <span style={{ fontFamily: FONT_MONO, fontSize, fontWeight: weight, color, whiteSpace: "pre", lineHeight: 1.15 }}>
      {parts.map((p, i) =>
        /^<(0x|U\+)/.test(p) ? (
          <span key={i} style={{ color: C.amber, fontSize: fontSize * 0.8 }}>
            {p}
          </span>
        ) : (
          Array.from(p).map((ch, j) => (
            <span key={`${i}-${j}`} style={MARKERS.has(ch) ? { color: onlyMarkers ? C.text : C.muted, fontWeight: 400 } : undefined}>
              {ch}
            </span>
          ))
        ),
      )}
    </span>
  );
};
