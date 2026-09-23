import type { Candidate } from "../data/types";
import { C, FONT_MONO } from "../utils/theme";
import { pct } from "../utils/format";

/**
 * Cumulative-probability meter. Each ranked candidate adds a segment equal to its
 * temperature_probability; segments turn grey once the running total has reached top_p.
 * `reveal` = how many candidates have been added so far (fractional values animate the last one).
 */
export const TopPNucleus: React.FC<{ candidates: Candidate[]; topP: number; reveal: number; width: number }> = ({
  candidates,
  topP,
  reveal,
  width,
}) => {
  const shown = Math.min(candidates.length, Math.floor(reveal));
  const partial = reveal - Math.floor(reveal);
  const drawing = Math.min(candidates.length, Math.ceil(reveal)); // readout follows the segment being drawn
  const last = drawing > 0 ? candidates[drawing - 1] : null;
  let x = 0;
  return (
    <div style={{ width, fontFamily: FONT_MONO }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 24, color: C.muted, marginBottom: 12 }}>
        <span>CUMULATIVE PROBABILITY</span>
        <span style={{ color: C.text, fontWeight: 700 }}>{last ? pct(last.cumulative_probability) : "0.0%"}</span>
      </div>
      <div style={{ position: "relative", height: 46, borderRadius: 12, background: "rgba(51,65,85,0.35)", border: `1.5px solid ${C.border}` }}>
        {candidates.map((c, i) => {
          const w = c.temperature_probability * width;
          const left = x;
          x += w;
          const vis = i < shown ? 1 : i === shown ? partial : 0;
          if (vis <= 0) return null;
          return (
            <div
              key={c.token_id}
              style={{
                position: "absolute",
                left,
                top: 0,
                bottom: 0,
                width: w * vis,
                background: c.inside_top_p ? (i % 2 ? C.teal : C.cyan) : "#475569",
                opacity: c.inside_top_p ? 0.9 : 0.8,
                borderRight: `2px solid ${C.bg}`,
              }}
            />
          );
        })}
        <div style={{ position: "absolute", left: topP * width - 2, top: -12, bottom: -12, width: 4, background: C.magenta, boxShadow: `0 0 16px ${C.magenta}` }} />
      </div>
      <div style={{ position: "relative", height: 34, marginTop: 8 }}>
        <span
          style={{
            position: "absolute",
            // centred under the threshold line, but kept inside the meter when top_p is close to 0 or 1
            left: Math.min(Math.max(topP * width - 90, 0), width - 180),
            width: 180,
            textAlign: "center",
            fontSize: 22,
            color: C.magenta,
            fontWeight: 700,
          }}
        >
          p = {topP.toFixed(2)}
        </span>
      </div>
    </div>
  );
};
