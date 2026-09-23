import { interpolate } from "remotion";
import type { TokenInfo } from "../data/types";
import { easeInOut } from "../utils/anim";
import { C } from "../utils/theme";
import { TokenChip } from "./TokenChip";

/** A token chip travelling from one absolute point to another while p goes 0 → 1. */
export const FlyingToken: React.FC<{ token: TokenInfo; from: [number, number]; to: [number, number]; p: number }> = ({ token, from, to, p }) => {
  if (p <= 0 || p >= 1) return null;
  const e = easeInOut(p);
  const x = interpolate(e, [0, 1], [from[0], to[0]]);
  const y = interpolate(e, [0, 1], [from[1], to[1]]) - Math.sin(e * Math.PI) * 80;
  return (
    <div style={{ position: "absolute", left: x, top: y, transform: `translate(-50%, -50%) scale(${1.1 - 0.35 * e})`, opacity: p > 0.85 ? (1 - p) / 0.15 : 1, zIndex: 10 }}>
      <TokenChip token={token} showId={false} color={C.magenta} size={40} style={{ boxShadow: `0 0 40px ${C.magenta}88`, background: "#2A1535" }} />
    </div>
  );
};
