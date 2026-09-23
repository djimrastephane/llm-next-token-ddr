import { Easing, interpolate } from "remotion";

export const easeOut = Easing.bezier(0.16, 1, 0.3, 1);
export const easeInOut = Easing.bezier(0.65, 0, 0.35, 1);

/** 0 → 1 between `start` and `start + len` frames, eased. */
export function prog(frame: number, start: number, len: number, easing = easeOut): number {
  return interpolate(frame, [start, start + len], [0, 1], {
    easing,
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
}

/** Scene opacity: fade in at start, fade out before `duration`. */
export function sceneFade(frame: number, duration: number, len = 14): number {
  return Math.min(prog(frame, 0, len, easeInOut), 1 - prog(frame, duration - len, len, easeInOut));
}

/** Slide-up + fade entrance style. */
export function rise(frame: number, start: number, len = 24, dist = 28): React.CSSProperties {
  const p = prog(frame, start, len);
  return { opacity: p, transform: `translateY(${(1 - p) * dist}px)` };
}
