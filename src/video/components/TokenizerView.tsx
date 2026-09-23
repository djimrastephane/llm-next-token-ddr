import type { InputToken } from "../data/types";
import { prog } from "../utils/anim";
import { C } from "../utils/theme";
import { TokenChip } from "./TokenChip";

/** Real input tokens as chips with their IDs, revealed one by one from `start`. */
export const TokenizerView: React.FC<{ tokens: InputToken[]; frame: number; start: number; every: number; width: number }> = ({
  tokens,
  frame,
  start,
  every,
  width,
}) => (
  <div style={{ width, display: "flex", flexWrap: "wrap", gap: 14, justifyContent: "center" }}>
    {tokens.map((t, i) => {
      const p = prog(frame, start + i * every, 16);
      return (
        <div key={t.position} style={{ opacity: p, transform: `scale(${0.85 + 0.15 * p})` }}>
          <TokenChip token={t} size={36} maxWidth={260} color={i % 2 ? C.purple : C.cyan} />
        </div>
      );
    })}
  </div>
);
