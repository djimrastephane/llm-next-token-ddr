import type { TokenInfo } from "../data/types";
import { C, FONT_MONO } from "../utils/theme";
import { TokenText } from "./TokenText";

export const TokenChip: React.FC<{
  token: TokenInfo;
  showId?: boolean;
  color?: string;
  size?: number;
  maxWidth?: number;
  style?: React.CSSProperties;
}> = ({ token, showId = true, color = C.cyan, size = 32, maxWidth = 300, style }) => (
  <div
    style={{
      display: "inline-flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "10px 16px 8px",
      borderRadius: 16,
      border: `1.5px solid ${color}55`,
      background: `linear-gradient(180deg, ${color}1f, ${color}0a)`,
      ...style,
    }}
  >
    <TokenText text={token.display_token} maxWidth={maxWidth} size={size} />
    {showId && (
      <span style={{ fontFamily: FONT_MONO, fontSize: 20, color, marginTop: 4, fontWeight: 500 }}>{token.token_id}</span>
    )}
  </div>
);
