"""SQLite schema initialisation — creates tables and inserts default settings."""

import aiosqlite
import config as cfg

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    slot_start TEXT NOT NULL,
    slot_end TEXT NOT NULL,
    slot_timestamp INTEGER NOT NULL,
    side TEXT,
    entry_price REAL,
    opposite_price REAL,
    outcome TEXT,
    is_win INTEGER,
    resolved_at TIMESTAMP,
    skipped INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    slot_start TEXT NOT NULL,
    slot_end TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    amount_usdc REAL NOT NULL,
    order_id TEXT,
    fill_price REAL,
    status TEXT DEFAULT 'pending',
    outcome TEXT,
    is_win INTEGER,
    pnl REAL,
    resolved_at TIMESTAMP,
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS redemptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    condition_id TEXT NOT NULL,
    outcome_index INTEGER NOT NULL,
    size REAL NOT NULL,
    title TEXT,
    tx_hash TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    gas_used INTEGER,
    dry_run INTEGER NOT NULL DEFAULT 0,
    resolved_at TIMESTAMP
);
"""

DEFAULT_SETTINGS = {
    "autotrade_enabled": "false",
    "trade_amount_usdc": str(cfg.TRADE_AMOUNT_USDC),
    "auto_redeem_enabled": "false",
}


async def init_db(db_path: str | None = None) -> None:
    """Create tables if they don't exist and seed default settings."""
    path = db_path or cfg.DB_PATH
    async with aiosqlite.connect(path) as db:
        await db.executescript(SCHEMA_SQL)
        for key, value in DEFAULT_SETTINGS.items():
            await db.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )
        await db.commit()
