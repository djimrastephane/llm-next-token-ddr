import type { Candidate } from "../data/types";
import { C, FONT_MONO } from "../utils/theme";
import { TokenText } from "./TokenText";

export type BarTone = "cyan" | "purple" | "excluded" | "selected";

const TONES: Record<BarTone, [string, string]> = {
  cyan: [C.cyan, C.teal],
  purple: [C.purple, C.magenta],
  excluded: ["#475569", "#334155"],
  selected: [C.magenta, C.purple],
};

export type BarSpec = {
  candidate: Candidate;
  value: number; // height, in probability units
  ghost?: number; // optional dashed outline (e.g. value before a transformation)
  tone: BarTone;
  label: string; // printed above the bar
  label2?: string; // smaller second line
  footer?: string; // below the token (rank, cumulative, EXCLUDED…)
  footerColor?: string;
  opacity?: number;
};

/** One vertical bar. `grow` (0..1) animates its height upward from zero. */
export const ProbabilityBar: React.FC<{ spec: BarSpec; slotW: number; barH: number; axis: number; grow: number; glow?: number }> = ({
  spec,
  slotW,
  barH,
  axis,
  grow,
  glow = 0,
}) => {
  const [top, bottom] = TONES[spec.tone];
  const h = Math.max(2, (spec.value / axis) * barH * grow);
  const bw = Math.min(104, slotW * 0.66);
  const ghostH = spec.ghost !== undefined ? (spec.ghost / axis) * barH : null;
  const muted = spec.tone === "excluded";
  return (
    <div style={{ width: slotW, display: "flex", flexDirection: "column", alignItems: "center", opacity: spec.opacity ?? 1 }}>
      <div style={{ height: barH + 78, width: "100%", position: "relative" }}>
        {ghostH !== null && (
          <div
            style={{
              position: "absolute",
              bottom: 0,
              left: (slotW - bw) / 2,
              width: bw,
              height: ghostH,
              border: `2.5px dashed ${C.muted}aa`,
              borderBottom: "none",
              borderRadius: "12px 12px 0 0",
            }}
          />
        )}
        <div
          style={{
            position: "absolute",
            bottom: 0,
            left: (slotW - bw) / 2,
            width: bw,
            height: h,
            borderRadius: "12px 12px 4px 4px",
            background: `linear-gradient(180deg, ${top}, ${bottom}${muted ? "" : "cc"})`,
            boxShadow: glow > 0 ? `0 0 ${40 * glow}px ${top}` : muted ? undefined : `0 0 18px ${top}40`,
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: h + 10,
            width: "100%",
            textAlign: "center",
            fontFamily: FONT_MONO,
            opacity: grow > 0.05 ? 1 : 0,
          }}
        >
          <div style={{ fontSize: slotW < 120 ? 24 : 28, fontWeight: 700, color: muted ? C.dim : C.text }}>{spec.label}</div>
          {spec.label2 && <div style={{ fontSize: 20, color: C.dim, marginTop: 2 }}>{spec.label2}</div>}
        </div>
      </div>
      <div style={{ height: 3, width: "100%", background: C.borderHi }} />
      <div style={{ height: 52, display: "flex", alignItems: "center", justifyContent: "center", width: slotW - 6 }}>
        <TokenText text={spec.candidate.display_token} maxWidth={slotW - 10} size={28} minSize={15} color={muted ? C.dim : C.text} />
      </div>
      <div style={{ fontFamily: FONT_MONO, fontSize: 20, color: spec.footerColor ?? C.dim, fontWeight: 700, height: 28, whiteSpace: "nowrap" }}>
        {spec.footer ?? `rank ${spec.candidate.rank}`}
      </div>
    </div>
  );
};
