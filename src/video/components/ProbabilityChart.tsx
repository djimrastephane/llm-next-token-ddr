import { C, FONT_MONO } from "../utils/theme";
import { pct } from "../utils/format";
import { ProbabilityBar, type BarSpec } from "./ProbabilityBar";

export const slotCenter = (i: number, n: number, width: number) => ((i + 0.5) * width) / n;

/** A row of vertical probability bars with a labelled axis. Values come from the trace via `bars`. */
export const ProbabilityChart: React.FC<{
  bars: BarSpec[];
  width: number;
  barH: number;
  axis: number;
  grow: (i: number) => number;
  glow?: (i: number) => number;
}> = ({ bars, width, barH, axis, grow, glow }) => {
  const slotW = width / bars.length;
  return (
    <div style={{ position: "relative", width }}>
      {[0.5, 1].map((g) => (
        <div
          key={g}
          style={{ position: "absolute", left: 0, right: 0, top: 78 + barH * (1 - g), borderTop: `1px dashed ${C.border}` }}
        >
          <span style={{ position: "absolute", right: 0, top: -26, fontFamily: FONT_MONO, fontSize: 18, color: C.dim }}>{pct(axis * g)}</span>
        </div>
      ))}
      <div style={{ display: "flex", position: "relative" }}>
        {bars.map((b, i) => (
          <ProbabilityBar key={b.candidate.token_id} spec={b} slotW={slotW} barH={barH} axis={axis} grow={grow(i)} glow={glow?.(i)} />
        ))}
      </div>
    </div>
  );
};
