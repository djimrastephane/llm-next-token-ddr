import { AbsoluteFill, useCurrentFrame } from "remotion";
import { Card } from "../components/Card";
import { SceneTitle } from "../components/SceneTitle";
import { Tag } from "../components/Tag";
import { TokenChip } from "../components/TokenChip";
import { TokenText } from "../components/TokenText";
import { comparison, meta, trace } from "../data/loadInferenceTrace";
import type { InferenceTrace } from "../data/types";
import { prog, rise, sceneFade } from "../utils/anim";
import { pct } from "../utils/format";
import { C, CONTENT_W, FONT_MONO, FONT_UI, PAD } from "../utils/theme";
import { sceneFrames } from "../utils/timeline";

const { greedy, diverge, identical } = comparison;

/** A token shown inline in a sentence, boxed so short tokens such as "." stay visible. */
const InlineToken: React.FC<{ text: string; color: string }> = ({ text, color }) => (
  <span style={{ display: "inline-block", border: `2px solid ${color}`, borderRadius: 10, padding: "0 10px", lineHeight: 1.25, background: `${color}1a` }}>
    <TokenText text={text} maxWidth={240} size={32} color={color} weight={800} />
  </span>
);
const lastSentence = trace.display_context.slice(trace.display_context.lastIndexOf(". ") + 2);
const CHIP_EVERY = 7;
const GREEDY_AT = 45;
const SAMPLING_AT = 70;
const DIVERGE_AT = 150;

/** One decoding run: its selected tokens as chips (with rank), then the resulting sentence. */
const Run: React.FC<{
  run: InferenceTrace;
  title: string;
  detail: string;
  color: string;
  start: number;
  f: number;
}> = ({ run, title, detail, color, start, f }) => (
  <Card style={{ ...rise(f, start), padding: "26px 28px", marginTop: 26 }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 16 }}>
      <span style={{ fontFamily: FONT_MONO, fontSize: 30, fontWeight: 800, color, letterSpacing: 1.5 }}>{title}</span>
      <span style={{ fontFamily: FONT_MONO, fontSize: 22, color: C.muted, textAlign: "right" }}>{detail}</span>
    </div>
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", gap: 12, marginTop: 20 }}>
      {run.steps.map((s, i) => {
        const p = prog(f, start + 12 + i * CHIP_EVERY, 14);
        const isDiverge = i === diverge && f > DIVERGE_AT;
        return (
          <div key={s.step} style={{ opacity: p, transform: `scale(${0.85 + 0.15 * p})`, textAlign: "center" }}>
            <TokenChip
              token={s.selected_token}
              showId={false}
              size={42}
              maxWidth={260}
              color={isDiverge ? C.magenta : color}
              style={isDiverge ? { boxShadow: `0 0 26px ${C.magenta}99` } : undefined}
            />
            <div style={{ fontFamily: FONT_MONO, fontSize: 19, color: s.selected_token.rank > 1 ? C.magenta : C.dim, marginTop: 6 }}>
              rank {s.selected_token.rank}
            </div>
          </div>
        );
      })}
    </div>
    <div style={{ ...rise(f, start + 14 + run.steps.length * CHIP_EVERY), fontSize: 38, color: C.text, marginTop: 20, lineHeight: 1.35 }}>
      …{lastSentence}
      <span style={{ color, fontWeight: 700 }}>{run.generated_text}</span>
    </div>
  </Card>
);

export const S8bCompare: React.FC = () => {
  const f = useCurrentFrame();
  const g = diverge !== null ? greedy.steps[diverge].selected_token : null;
  const s = diverge !== null ? trace.steps[diverge].selected_token : null;
  const hasNumbers = /\d/.test(greedy.generated_text + trace.generated_text);
  return (
    <AbsoluteFill style={{ opacity: sceneFade(f, sceneFrames("compare")), fontFamily: FONT_UI }}>
      <div style={{ position: "absolute", top: 150, left: PAD, width: CONTENT_W }}>
        <SceneTitle
          kicker="08 · GREEDY vs SAMPLING"
          title="Same model, same context, two decoding rules"
          subtitle={`${meta.model_display_name} · both runs captured from the same DDR text`}
        />
        <Card style={{ ...rise(f, 20), padding: "18px 26px", marginTop: 30, fontSize: 32, color: C.muted }}>
          <span style={{ fontFamily: FONT_MONO, fontSize: 18, color: C.dim, marginRight: 14 }}>CONTEXT</span>…{lastSentence}
        </Card>
        <Run run={greedy} title="GREEDY" detail="always rank 1 · no randomness" color={C.cyan} start={GREEDY_AT} f={f} />
        <Run
          run={trace}
          title="TOP-P SAMPLING"
          detail={`T ${meta.temperature?.toFixed(2)} · p ${meta.top_p?.toFixed(2)} · seed ${meta.seed}`}
          color={C.purple}
          start={SAMPLING_AT}
          f={f}
        />
        <Card glow={C.magenta} style={{ ...rise(f, DIVERGE_AT), marginTop: 30, padding: "24px 28px", fontSize: 31, lineHeight: 1.5, color: C.muted }}>
          {identical || !g || !s ? (
            <>Here both rules produced the same text. That can happen: sampling often draws the top token.</>
          ) : (
            <>
              <b style={{ color: C.text }}>They split at step t = {diverge! + 1}.</b> Greedy took{" "}
              <InlineToken text={g.display_token} color={C.cyan} /> (rank {g.rank}); sampling drew{" "}
              <InlineToken text={s.display_token} color={C.magenta} /> (rank {s.rank}, {pct(s.sampling_probability ?? 0)} chance).
              Neither is &ldquo;the right answer&rdquo;: both are real outputs of the same model.
            </>
          )}
        </Card>
        {hasNumbers && (
          <div style={{ ...rise(f, DIVERGE_AT + 30), marginTop: 26 }}>
            <Tag style={{ fontSize: 22, whiteSpace: "normal", lineHeight: 1.45 }}>
              NUMBERS IN THE GENERATED TEXT ARE MODEL OUTPUT, NOT A HYDRAULICS CALCULATION OR MEASUREMENT
            </Tag>
          </div>
        )}
      </div>
    </AbsoluteFill>
  );
};
