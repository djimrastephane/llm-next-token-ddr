import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";

const inter = loadInter("normal", { weights: ["400", "500", "600", "700", "800"], subsets: ["latin"] });
const mono = loadMono("normal", { weights: ["400", "500", "700"], subsets: ["latin"] });

export const FONT_UI = inter.fontFamily;
export const FONT_MONO = mono.fontFamily;

export const C = {
  bg: "#0B0F19",
  surface: "rgba(17, 24, 39, 0.82)",
  surfaceHi: "rgba(30, 41, 59, 0.85)",
  border: "rgba(148, 163, 184, 0.18)",
  borderHi: "rgba(148, 163, 184, 0.35)",
  text: "#E6EAF2",
  muted: "#94A3B8",
  dim: "#64748B",
  cyan: "#22D3EE",
  teal: "#2DD4BF",
  purple: "#A78BFA",
  magenta: "#E879F9",
  excluded: "#334155",
  amber: "#FBBF24",
  red: "#FF5F57",
  yellow: "#FEBC2E",
  green: "#28C840",
} as const;

export const W = 1080;
export const H = 1920;
export const FPS = 60;
export const PAD = 48;
export const CONTENT_W = W - PAD * 2;
