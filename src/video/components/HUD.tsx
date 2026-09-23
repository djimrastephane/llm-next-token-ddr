import { meta } from "../data/loadInferenceTrace";
import { C, CONTENT_W, FONT_MONO } from "../utils/theme";
import { MetricCard } from "./MetricCard";

/** LLM settings only. Oil & Gas facts live in ContextDisplay, never here. */
export const HUD: React.FC<{ step: number; stepFlash?: number }> = ({ step, stepFlash = 0 }) => (
  <div style={{ width: CONTENT_W }}>
    <div style={{ fontFamily: FONT_MONO, fontSize: 18, color: C.dim, letterSpacing: 2, fontWeight: 700, marginBottom: 10 }}>
      LLM SETTINGS · FROM INFERENCE TRACE
    </div>
    <div style={{ display: "flex", gap: 12 }}>
      <MetricCard label="MODEL" value={meta.model_display_name} accent={C.cyan} flex={2.6} />
      <MetricCard label="DEVICE" value={`${meta.device} · local`} flex={1.4} />
    </div>
    <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
      <MetricCard label="TOKEN STEP" value={`t = ${step}`} accent={C.magenta} highlight={stepFlash} />
      <MetricCard label="TEMPERATURE" value={meta.temperature?.toFixed(2) ?? "—"} />
      <MetricCard label="TOP-P" value={meta.top_p?.toFixed(2) ?? "—"} />
      <MetricCard label="SEED" value={meta.seed} flex={0.8} />
    </div>
  </div>
);
