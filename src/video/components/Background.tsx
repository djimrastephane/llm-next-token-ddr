import { AbsoluteFill } from "remotion";
import { C } from "../utils/theme";

export const Background: React.FC = () => (
  <AbsoluteFill style={{ backgroundColor: C.bg }}>
    <AbsoluteFill
      style={{
        backgroundImage:
          "linear-gradient(rgba(148,163,184,0.045) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.045) 1px, transparent 1px)",
        backgroundSize: "54px 54px",
      }}
    />
    <AbsoluteFill
      style={{
        background:
          "radial-gradient(900px 700px at 15% 8%, rgba(34,211,238,0.10), transparent 60%), radial-gradient(900px 800px at 95% 70%, rgba(167,139,250,0.10), transparent 60%)",
      }}
    />
  </AbsoluteFill>
);
