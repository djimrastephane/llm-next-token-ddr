import { C } from "../utils/theme";

export const Card: React.FC<{ children: React.ReactNode; style?: React.CSSProperties; glow?: string }> = ({
  children,
  style,
  glow,
}) => (
  <div
    style={{
      background: C.surface,
      border: `1.5px solid ${glow ?? C.border}`,
      borderRadius: 28,
      boxShadow: glow ? `0 0 42px ${glow}33, inset 0 0 24px ${glow}14` : "0 10px 40px rgba(0,0,0,0.35)",
      ...style,
    }}
  >
    {children}
  </div>
);
