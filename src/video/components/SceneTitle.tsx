import { useCurrentFrame } from "remotion";
import { rise } from "../utils/anim";
import { C, FONT_MONO, FONT_UI } from "../utils/theme";

export const SceneTitle: React.FC<{ kicker: string; title: React.ReactNode; subtitle?: React.ReactNode; delay?: number }> = ({
  kicker,
  title,
  subtitle,
  delay = 0,
}) => {
  const f = useCurrentFrame();
  return (
    <div style={{ fontFamily: FONT_UI }}>
      <div style={{ ...rise(f, delay), fontFamily: FONT_MONO, color: C.cyan, fontSize: 26, fontWeight: 700, letterSpacing: 3 }}>
        {kicker}
      </div>
      <div style={{ ...rise(f, delay + 5), color: C.text, fontSize: 64, fontWeight: 800, lineHeight: 1.05, marginTop: 10, letterSpacing: -1 }}>
        {title}
      </div>
      {subtitle && (
        <div style={{ ...rise(f, delay + 10), color: C.muted, fontSize: 32, fontWeight: 500, marginTop: 14, lineHeight: 1.3 }}>
          {subtitle}
        </div>
      )}
    </div>
  );
};
