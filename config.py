"""AutoPoly configuration — loads from environment variables with sensible defaults."""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------
POLYMARKET_PRIVATE_KEY: str | None = os.getenv("POLYMARKET_PRIVATE_KEY")
POLYMARKET_FUNDER_ADDRESS: str | None = os.getenv("POLYMARKET_FUNDER_ADDRESS")
POLYMARKET_SIGNATURE_TYPE: int = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "2"))

CLOB_HOST: str = "https://clob.polymarket.com"
GAMMA_API_HOST: str = "https://gamma-api.polymarket.com"
CHAIN_ID: int = 137

# ---------------------------------------------------------------------------
# Polygon RPC (required for on-chain redemptions via web3.py)
# ---------------------------------------------------------------------------
POLYGON_RPC_URL: str = os.getenv(
    "POLYGON_RPC_URL",
    "https://polygon-rpc.com",  # public fallback — consider Alchemy/Infura for production
)

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str | None = os.getenv("TELEGRAM_CHAT_ID")

# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------
TRADE_AMOUNT_USDC: float = float(os.getenv("TRADE_AMOUNT_USDC", "1.0"))

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: str = os.getenv("DB_PATH", "autopoly.db")

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
SIGNAL_THRESHOLD: float = 0.51
SIGNAL_LEAD_TIME: int = 85  # seconds before slot end to check signal

# ---------------------------------------------------------------------------
# ADX Filter
# ---------------------------------------------------------------------------
ADX_LENGTH: int = 14                # ADX period (Wilder's smoothing)
ADX_CANDLE_COUNT: int = 300         # 5-min candles to fetch from Coinbase (max Coinbase allows; needed for ADX warm-up)
COINBASE_CANDLE_URL: str = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

# ---------------------------------------------------------------------------
# Auto-Redeem
# ---------------------------------------------------------------------------
# Scheduler interval (minutes) between automatic redemption scans.
AUTO_REDEEM_INTERVAL_MINUTES: int = int(os.getenv("AUTO_REDEEM_INTERVAL_MINUTES", "30"))
