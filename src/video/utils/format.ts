// Presentation-only formatting. Never changes a value, only how it is printed.

export function pct(p: number): string {
  const v = p * 100;
  if (v >= 99.95 && p < 1) return ">99.9%";
  if (v >= 1) return `${v.toFixed(1)}%`;
  if (v >= 0.01) return `${v.toFixed(2)}%`;
  return "<0.01%";
}

export const num = (x: number, digits = 2) => x.toFixed(digits);

export const int = (x: number) => x.toLocaleString("en-US");

/** Font size that fits `text` into `maxWidth` at monospace advance ~0.6em, clamped to [min, max]. */
export function fitMono(text: string, maxWidth: number, max: number, min = 16): number {
  const chars = Array.from(text).length || 1;
  return Math.max(min, Math.min(max, maxWidth / (chars * 0.62)));
}

/** Middle-ellipsis for token strings too long even at the minimum font size. */
export function clampToken(text: string, maxWidth: number, min = 16): string {
  const chars = Array.from(text);
  const capacity = Math.floor(maxWidth / (min * 0.62));
  if (chars.length <= capacity) return text;
  const keep = Math.max(2, capacity - 1);
  const head = Math.ceil(keep / 2);
  return chars.slice(0, head).join("") + "…" + chars.slice(chars.length - (keep - head)).join("");
}

/** Round an axis maximum up to a clean 10% step so bars use the chart height. */
export function axisMax(maxValue: number): number {
  return Math.min(1, Math.max(0.1, Math.ceil(maxValue * 10 - 1e-9) / 10));
}
