// QA helper: bundle once, render several stills. Usage: node scripts/render_stills.mjs 250 560 1200 ...
import { bundle } from "@remotion/bundler";
import { renderStill, selectComposition } from "@remotion/renderer";
import path from "node:path";

const frames = process.argv.slice(2).map(Number);
const serveUrl = await bundle({ entryPoint: path.resolve("src/video/index.ts") });
const composition = await selectComposition({ serveUrl, id: "NextTokenDDR" });
for (const frame of frames) {
  const output = path.resolve(`out/stills/frame_${String(frame).padStart(5, "0")}.png`);
  await renderStill({ serveUrl, composition, frame, output });
  console.log(output);
}
