import { C } from "../utils/theme";

/**
 * Overhead selector claw (CONCEPTUAL VIEW). Positioned by its horizontal centre `x`;
 * `drop` is the cable length, `grip` 0 = open, 1 = closed on the token.
 */
export const SelectionClaw: React.FC<{ x: number; drop: number; grip: number; glow: number; color?: string }> = ({
  x,
  drop,
  grip,
  glow,
  color = C.magenta,
}) => {
  const open = 28 - 22 * grip; // prong angle in degrees
  const w = 140;
  return (
    <svg
      width={w}
      height={drop + 110}
      style={{ position: "absolute", left: x - w / 2, top: 0, overflow: "visible", filter: `drop-shadow(0 0 ${6 + 18 * glow}px ${color})` }}
    >
      <line x1={w / 2} y1={0} x2={w / 2} y2={drop} stroke={C.muted} strokeWidth={3} />
      <rect x={w / 2 - 34} y={drop} width={68} height={26} rx={9} fill="#1E293B" stroke={color} strokeWidth={3} />
      <circle cx={w / 2} cy={drop + 13} r={5} fill={color} />
      {[-1, 1].map((side) => (
        <g key={side} transform={`rotate(${side * open} ${w / 2 + side * 22} ${drop + 26})`}>
          <path
            d={`M ${w / 2 + side * 22} ${drop + 26} L ${w / 2 + side * 30} ${drop + 66} L ${w / 2 + side * 14} ${drop + 92}`}
            fill="none"
            stroke={color}
            strokeWidth={7}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </g>
      ))}
    </svg>
  );
};
