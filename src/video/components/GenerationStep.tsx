import type { Step } from "../data/types";
import { meta } from "../data/loadInferenceTrace";
import { prog } from "../utils/anim";
import { clawPath, clawSlot } from "../utils/claw";
import { axisMax, pct } from "../utils/format";
import { C, FONT_MONO } from "../utils/theme";
import type { BarSpec } from "./ProbabilityBar";
import { ProbabilityChart, slotCenter } from "./ProbabilityChart";
import { SelectionClaw } from "./SelectionClaw";
import { Tag } from "./Tag";

/**
 * One fast autoregressive step: sampling probabilities → claw locks onto the recorded selection.
 * Shows the nucleus plus up to two excluded tokens (max 8 bars).
 */
export const GenerationStep: React.FC<{ step: Step; frame: number; width: number; barH: number }> = ({ step, frame, width, barH }) => {
  const n = step.nucleus_size ?? 1;
  const cands = step.top_candidates.slice(0, Math.min(8, Math.max(n + 2, step.selected_token.rank)));
  const shownEligible = cands.filter((c) => c.inside_top_p).length;
  const selIdx = cands.findIndex((c) => c.selected);
  const bars: BarSpec[] = cands.map((c) => ({
    candidate: c,
    value: c.sampling_probability ?? 0, // 0 outside the nucleus: one quantity per chart
    tone: c.selected && frame > 60 ? "selected" : c.inside_top_p ? "cyan" : "excluded",
    label: c.inside_top_p ? pct(c.sampling_probability ?? 0) : "0%",
    footer: c.inside_top_p ? `rank ${c.rank}` : "EXCLUDED",
  }));
  const axis = axisMax(Math.max(...bars.map((b) => b.value)));
  const path = clawPath(shownEligible, selIdx, 1);
  const slot = clawSlot(path, prog(frame, 14, 44));
  const grip = prog(frame, 56, 10);
  const lock = prog(frame, 60, 14);
  return (
    <div style={{ width }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", fontFamily: FONT_MONO, marginBottom: 6 }}>
        <span style={{ fontSize: 34, fontWeight: 800, color: C.magenta }}>STEP t = {step.step}</span>
        <span style={{ fontSize: 21, color: C.muted }}>
          nucleus: {n} token{n === 1 ? "" : "s"}
          {n > shownEligible ? ` (${shownEligible} shown)` : ""} · p = {meta.top_p?.toFixed(2)}
        </span>
      </div>
      <div style={{ position: "relative", paddingTop: 150 }}>
        <div style={{ position: "absolute", inset: 0, opacity: prog(frame, 8, 10) }}>
          <SelectionClaw x={slotCenter(slot, cands.length, width)} drop={50} grip={grip} glow={lock} />
        </div>
        <ProbabilityChart bars={bars} width={width} barH={barH} axis={axis} grow={(i) => prog(frame, i * 2, 18)} glow={(i) => (i === selIdx ? lock : 0)} />
      </div>
      <div style={{ height: 64, marginTop: 10, textAlign: "center", opacity: lock }}>
        {step.selected_token.rank > 1 ? (
          <Tag color={C.magenta} style={{ fontSize: 24 }}>
            RANK {step.selected_token.rank} SELECTED · NOT THE MOST LIKELY TOKEN
          </Tag>
        ) : n === 1 ? (
          <Tag color={C.teal} style={{ fontSize: 24 }}>
            ONLY ONE TOKEN IN THE NUCLEUS · NO REAL CHOICE
          </Tag>
        ) : null}
      </div>
    </div>
  );
};
