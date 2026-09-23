import { Composition } from "remotion";
import { NextTokenVideo } from "./compositions/NextTokenVideo";
import { FPS, H, W } from "./utils/theme";
import { TOTAL_FRAMES } from "./utils/timeline";

export const Root: React.FC = () => (
  <Composition id="NextTokenDDR" component={NextTokenVideo} durationInFrames={TOTAL_FRAMES} fps={FPS} width={W} height={H} />
);
