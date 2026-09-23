import { AbsoluteFill, Sequence, useCurrentFrame } from "remotion";
import { Card } from "../components/Card";
import { FlyingToken } from "../components/FlyingToken";
import { GenerationStep } from "../components/GenerationStep";
import { SceneTitle } from "../components/SceneTitle";
import { TokenChip } from "../components/TokenChip";
import { TokenText } from "../components/TokenText";
import { trace } from "../data/loadInferenceTrace";
import { prog, rise, sceneFade } from "../utils/anim";
import { pct } from "../utils/format";
import { C, CONTENT_W, FONT_MONO, FONT_UI, W } from "../utils/theme";
import { LOOP_EACH, LOOP_FIRST, appendedCount, sceneFrames, sceneStart } from "../utils/timeline";
import { STAGE_LEFT, STAGE_TOP } from "./layout";

const CONTEXT_CARD: [number, number] = [W / 2, 1735]; // where appended tokens land (DDR card)
/** Last ~n characters, starting at a word boundary. */
const tail = (s: string, n = 30) => {
  if (s.length <= n) return s;
  const cut = s.indexOf(" ", s.length - n);
  return "…" + s.slice(cut >= 0 ? cut : s.length - n);
};

const Equation: React.FC<{ f: number }> = ({ f }) => {
  const s = trace.steps[0];
  const box = (label: string, text: string, start: number, color: string) => (
    <Card style={{ ...rise(f, start), padding: "20px 28px" }}>
      <div style={{ fontFamily: FONT_MONO, fontSize: 22, color, fontWeight: 800, letterSpacing: 1.5 }}>{label}</div>
      <div style={{ fontSize: 40, color: C.text, marginTop: 6, whiteSpace: "pre" }}>{text}</div>
    </Card>
  );
  return (
    <div style={{ opacity: 1 - prog(f, LOOP_FIRST - 16, 14) }}>
      {box("CONTEXT t", tail(s.context_before), 8, C.cyan)}
      <div style={{ ...rise(f, 20), display: "flex", alignItems: "center", justifyContent: "center", gap: 24, margin: "30px 0" }}>
        <span style={{ fontSize: 48, color: C.dim }}>+</span>
        <div style={{ opacity: f < 60 ? 1 : 0.25 }}>
          <TokenChip token={s.selected_token} showId={false} color={C.magenta} size={52} />
        </div>
      </div>
      <div style={{ ...rise(f, 30), textAlign: "center", fontSize: 48, color: C.dim, marginBottom: 30 }}>↓</div>
      {box("CONTEXT t+1", tail(s.context_after), 40, C.magenta)}
      <FlyingToken token={s.selected_token} from={[W / 2, STAGE_TOP + 420]} to={CONTEXT_CARD} p={prog(f, 60, 35)} />
    </div>
  );
};

const StepLog: React.FC<{ count: number }> = ({ count }) => (
  <div style={{ fontFamily: FONT_MONO }}>
    {trace.steps.slice(0, count).map((s) => (
      <div key={s.step} style={{ display: "flex", alignItems: "center", height: 48, borderBottom: `1px solid ${C.border}`, fontSize: 24, color: C.muted, gap: 18 }}>
        <span style={{ width: 90, color: C.magenta, fontWeight: 700 }}>t = {s.step}</span>
        <span style={{ width: 260 }}>
          <TokenText text={s.selected_token.display_token} maxWidth={250} size={26} />
        </span>
        <span style={{ width: 150 }}>rank {s.selected_token.rank}</span>
        <span>{pct(s.selected_token.sampling_probability ?? s.selected_token.model_probability)} sampling</span>
      </div>
    ))}
  </div>
);

export const S8Loop: React.FC = () => {
  const f = useCurrentFrame();
  const loop0 = sceneStart("loop");
  const later = trace.steps.slice(1);
  const chartTop = STAGE_TOP + 200;
  return (
    <AbsoluteFill style={{ opacity: sceneFade(f, sceneFrames("loop")), fontFamily: FONT_UI }}>
      <div style={{ position: "absolute", top: STAGE_TOP, left: STAGE_LEFT, width: CONTENT_W }}>
        <SceneTitle kicker="07 · AUTOREGRESSIVE LOOP" title="Append the token. Repeat." />
        <div style={{ marginTop: 30, position: "relative", height: 820 }}>
          <Sequence durationInFrames={LOOP_FIRST} layout="none">
            <Equation f={f} />
          </Sequence>
          {later.map((s, i) => (
            <Sequence key={s.step} from={LOOP_FIRST + i * LOOP_EACH} durationInFrames={LOOP_EACH} layout="none">
              <StepFrame step={i + 1} />
            </Sequence>
          ))}
        </div>
        <div style={{ opacity: prog(f, 100, 20) }}>
          <StepLog count={appendedCount(loop0 + f)} />
        </div>
      </div>
      {later.map((s, i) => {
        const local = f - LOOP_FIRST - i * LOOP_EACH;
        const selIdx = s.top_candidates.slice(0, 8).findIndex((c) => c.selected);
        const shown = Math.min(8, Math.max((s.nucleus_size ?? 1) + 2, s.selected_token.rank));
        const x = STAGE_LEFT + ((selIdx + 0.5) * CONTENT_W) / shown;
        return <FlyingToken key={s.step} token={s.selected_token} from={[x, chartTop + 420]} to={CONTEXT_CARD} p={prog(local, 66, 22)} />;
      })}
    </AbsoluteFill>
  );
};

const StepFrame: React.FC<{ step: number }> = ({ step }) => {
  const f = useCurrentFrame();
  return (
    <div style={{ opacity: Math.min(prog(f, 0, 8), 1 - prog(f, LOOP_EACH - 8, 8)) }}>
      <GenerationStep step={trace.steps[step]} frame={f} width={CONTENT_W} barH={300} />
    </div>
  );
};
