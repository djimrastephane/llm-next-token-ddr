import { interpolate } from "remotion";
import { easeInOut } from "./anim";

/** Slot indices the claw visits: sweep right, back left, then settle on `target`. Decorative only. */
export function clawPath(n: number, target: number, sweeps = 2): number[] {
  if (n <= 1) return [0];
  const path: number[] = [0];
  const push = (to: number) => {
    let cur = path[path.length - 1];
    while (cur !== to) {
      cur += Math.sign(to - cur);
      path.push(cur);
    }
  };
  push(n - 1);
  if (sweeps > 1) push(0);
  push(target);
  return path;
}

/** Horizontal claw position (fractional slot index) at progress 0..1; later moves are slower (decelerating). */
export function clawSlot(path: number[], p: number): number {
  if (path.length === 1) return path[0];
  const durs = path.slice(1).map((_, i) => 1 + i * 0.35);
  const total = durs.reduce((a, b) => a + b, 0);
  let t = p * total;
  for (let i = 0; i < durs.length; i++) {
    if (t <= durs[i]) return interpolate(t / durs[i], [0, 1], [path[i], path[i + 1]], { easing: easeInOut });
    t -= durs[i];
  }
  return path[path.length - 1];
}
