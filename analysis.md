# AutoPoly - Comprehensive Codebase Analysis

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Execution Flow](#execution-flow)
- [File-by-File Analysis](#file-by-file-analysis)
- [Trading Strategy Deep Dive](#trading-strategy-deep-dive)
- [Risk Management & Filters](#risk-management--filters)
- [Position Sizing](#position-sizing)
- [Key Parameters & Config](#key-parameters--config)
- [Data Model](#data-model)

---

## Architecture Overview

AutoPoly is an automated trading bot for **Polymarket's BTC Up/Down 5-minute binary options markets**. It operates on Polygon mainnet, using the Polymarket CLOB (Central Limit Order Book) for order execution. The bot runs as a Telegram-controlled daemon that checks market prices every 5 minutes, generates trading signals based on 6-candle BTC-USD pattern matching, and optionally executes trades.

### Layer Structure

```
main.py (entry point)
  |
  +-- bot/          (Telegram UI layer: handlers, keyboards, formatters, auth)
  +-- core/         (Trading engine: strategy, execution, scheduling, resolution)
  +-- polymarket/   (Polymarket API layer: CLOB client, markets, account)
  +-- db/           (Persistence: SQLite schema, CRUD, analytics)
  +-- config.py     (Environment configuration)
```

---

## Execution Flow

```
[STARTUP]
  1. main.py: Validate environment variables (required keys)
  2. Initialize SQLite database (create tables, seed defaults)
  3. Initialize Polymarket CLOB client (derive API credentials from private key)
  4. Build Telegram Application with post_init hook
  5. In post_init:
     a. start_scheduler() - creates APScheduler, starts reconciler + auto-redeem jobs
     b. recover_unresolved() - schedules immediate resolution for any unresolved signals
     c. Register bot commands
  6. handlers.set_poly_client(poly_client) - inject trading client
  7. Begin Telegram polling (blocks)

[5-MINUTE TRADING LOOP] (runs via APScheduler, synced to slot boundaries)
  1. _check_and_trade() fires at T-85s (85 seconds before current slot N ends)
  2. strategy.check_signal() delegates to PatternStrategy:
     a. Fetch the 10 most recently confirmed-closed 5-min BTC-USD candles
        from Coinbase (dropping the still-open candle at the tail for safety)
     b. Build 6-char pattern string from candles N-1..N-6
        Format: [N-1][N-2][N-3][N-4][N-5][N-6], U=up D=down
     c. Look up pattern in PATTERN_TABLE (22 known patterns)
     d. If match: fetch Polymarket slot prices, return full signal dict
     e. If no match: return skip dict (no trade placed)
  3. Insert signal into DB
  4. TradeManager.check() — always returns allowed=True (passthrough, no filters)
  5. If autotrade/demotrade enabled:
     a. Create trade record in DB (status=pending)
     b. Place FOK order via trader.place_fok_order_with_retry()
     c. Up to 3 retry attempts with exponential backoff
     d. Duplicate guard + time fence safety checks
  6. Schedule resolution job for when slot N+1 ends (+30s buffer)
  7. _schedule_next() schedules next T-85s check

[RESOLUTION] (fires ~5.5 min after signal, after slot N+1 ends)
  1. resolver.resolve_slot() polls Coinbase for the 5-min candle covering the slot
  2. Winner: close >= open => "Up", else "Down"
  3. Compare winner vs traded side => is_win
  4. P&L calculation:
     - Win pnl = amount_usdc * (1/entry_price - 1)
     - Loss pnl = -amount_usdc
  5. Update signal and trade records in DB
  6. Send Telegram notifications
  7. For demo: credit bankroll on win

[RECONCILIATION] (every 5 minutes)
  - Check pending_queue for unresolved slots
  - Attempt resolution once per pending item
  - Remove resolved items, keep unresolved for next cycle

[AUTO-REDEEM] (every 5 minutes, or custom interval)
  - Scan Polymarket Data API for winning positions in wallet
  - Call CTF.redeemPositions() on Polygon to collect USDC.e
  - Record results in redemptions table
  - Send Telegram summary
```

---

## File-by-File Analysis

### ROOT LEVEL

#### main.py
**Purpose**: Application entry point, orchestrates startup sequence.

**Key Logic**:
1. `_validate_config()` - checks 4 required env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, POLYMARKET_PRIVATE_KEY, POLYMARKET_FUNDER_ADDRESS. Warns (non-blocking) about missing POLYGON_RPC_URL.
2. `asyncio.run(init_db())` - synchronous DB init.
3. `PolymarketClient(cfg)` - initializes CLOB client with private key, derives L2 API credentials.
4. Builds Telegram Application with `post_init` hook that:
   - Calls `start_scheduler(application, poly_client)` - MUST happen first
   - Calls `recover_unresolved()` second (depends on SCHEDULER global)
   - Registers bot commands
5. Injects poly_client and start_time into handlers module.
6. Calls `application.run_polling()` - blocks until stopped.

**Connections**: Imports from bot, core, db, polymarket, config. Serves as the glue layer.

#### config.py
**Purpose**: Single source of truth for all configuration, loaded from environment variables.

**Trading Parameters**:
- `TRADE_AMOUNT_USDC` = 1.0 (default)
- `SIGNAL_LEAD_TIME` = 85 seconds
- `FOK_MAX_RETRIES` = 3
- `FOK_RETRY_DELAY_BASE` = 2.0s (exponential: 2s, 4s, max 5s)
- `FOK_RETRY_DELAY_MAX` = 5.0s
- `FOK_SLOT_CUTOFF_SECONDS` = 30 (abort if less than 30s until slot end)
- `COINBASE_CANDLE_URL` = Coinbase API endpoint (used by pattern strategy)

**Polymarket**:
- `CLOB_HOST` = https://clob.polymarket.com
- `GAMMA_API_HOST` = https://gamma-api.polymarket.com
- `CHAIN_ID` = 137 (Polygon mainnet)
- `POLYGON_RPC_URL` = https://polygon-rpc.com (fallback)

**Auto-Redeem**:
- `AUTO_REDEEM_INTERVAL_MINUTES` = 5

**Connections**: Imported by every module.

#### README.md
**Purpose**: Project documentation, deployment instructions, strategy explanation.

#### Procfile / railway.toml
**Deployment**: Defines `worker: python main.py` for Railway.

#### requirements.txt
```
py-clob-client>=0.34.0
python-telegram-bot>=20.0
httpx>=0.25.0
apscheduler>=3.10.0
python-dotenv>=1.0.0
aiosqlite>=0.19.0
openpyxl>=3.1.0
web3>=6.0.0
```

---

### bot/ (Telegram Interface Layer)

#### bot/handlers.py
**Purpose**: All Telegram command and callback query handlers.

**Commands**:
- `/start` - Main menu with inline keyboard
- `/status` - Bot health, balance, autotrade state, open positions, uptime, last signal
- `/signals` - Signal performance dashboard (win rate, streaks, recent signals)
- `/trades` - Trade P&L dashboard (net P&L, ROI, streaks, recent trades)
- `/settings` - Toggle autotrade, auto-redeem, demo mode; change trade amount; set demo bankroll
- `/help` - Command reference + strategy explanation
- `/redeem` - Manual redemption (dry-run preview, then confirm)
- `/redemptions` - Redemption history dashboard
- `/demo` - Demo trading performance dashboard

**Callback Router**: Routes inline keyboard button presses. Key callbacks:
- `toggle_autotrade`, `toggle_auto_redeem`, `toggle_demo_trade` - Toggle settings in DB
- `change_amount` - Prompts user for numeric input
- `redeem_confirm` - Executes actual redemptions after dry-run preview
- `set_demo_bankroll` / `reset_demo_bankroll` - Manage virtual bankroll
- Signal/trade filter buttons (Last 10/50/All Time)
- `download_csv` / `download_xlsx` - Export all signals (columns: id, slot_start, side, entry_price, is_win)

**Key Patterns**:
- `@auth_check` decorator on every command
- `_safe_edit()` helper silently ignores "Message is not modified" errors
- `text_handler` catches free-text input for amount/bankroll changes
- Global `_poly_client` and `_start_time` injected at startup
- Unhandled error handler sends traceback to Telegram

#### bot/keyboards.py
**Purpose**: Inline keyboard layout generation.

**Layouts**: main_menu (2-column grid), settings_keyboard (toggle buttons + inputs), filter rows, redeem confirm/cancel, demo filters.

#### bot/formatters.py
**Purpose**: All message formatting with box-drawing characters and emojis.

**Live notifications** (sent by scheduler):
- `format_signal()` - Signal fired (includes pattern string)
- `format_skip()` - No pattern match, slot skipped
- `format_signal_resolution()` / `format_trade_resolution()` / `format_demo_resolution()` - Results
- `format_trade_filled()` / `format_trade_unmatched()` / `format_trade_aborted()` - Order outcomes
- `format_trade_retrying()` - Inline retry status
- `format_auto_redeem_notification()` / `format_error_alert()`

**Dashboards** (on command):
- `format_status()` / `format_signal_stats()` / `format_trade_stats()` / `format_demo_stats()`
- `format_redeem_preview()` / `format_redeem_results()` / `format_redemption_history()`

#### bot/middleware.py
**Purpose**: Single-chat authentication. `@auth_check` compares chat_id against TELEGRAM_CHAT_ID. Unauthorized updates silently dropped.

---

### core/ (Trading Engine)

#### core/scheduler.py
**Purpose**: Manages the 5-minute trading cycle via APScheduler.

**Key Functions**:
- `start_scheduler(tg_app, poly_client)` - Creates AsyncIOScheduler, starts reconciler (every 5 min), auto-redeem (every 5 min), schedules first check
- `_next_check_time()` - Calculates next T-85s check. Slots align to :00, :05, :10... Check fires at slot_start + 215 seconds
- `_check_and_trade()` - **Core loop body**: signal check, insertion, filtering, trade execution, resolution scheduling
- `_resolve_and_notify()` - Polls Coinbase for resolution, updates DB, calculates P&L
- `_reconcile_pending()` - Retries resolution for persistent queue items every 5 min
- `_auto_redeem_job()` - Runs auto-redemption scan if enabled
- `recover_unresolved()` - On startup, reschedules any unresolved signals

#### core/strategy.py
**Purpose**: Strategy orchestrator — thin wrapper around the active strategy.

**Logic**: Loads the configured strategy class (default: `PatternStrategy`) via the registry in `core/strategies/__init__.py` and delegates `check_signal()` to it. The singleton is cached in a module-level variable.

#### core/strategies/base.py
**Purpose**: Abstract base class for all strategies.

Defines the `BaseStrategy` ABC with a single async method: `check_signal() -> dict | None`.

#### core/strategies/pattern_strategy.py
**Purpose**: 6-candle BTC-USD pattern matching strategy.

**Algorithm**:
1. Fetch 10 confirmed-closed 5-min BTC-USD candles from Coinbase.
   - Drops the most-recent candle in the API response as a safety measure
     against a still-open candle at the tail (called at T-85s).
   - Returns exactly the requested count from the confirmed-closed set.
2. Build 6-char pattern string from candles N-1..N-6:
   - Character order (left to right): [N-1][N-2][N-3][N-4][N-5][N-6]
   - U = close >= open (up), D = close < open (down)
3. Look up pattern in PATTERN_TABLE (22 known profitable patterns).
4. On match: fetch Polymarket slot N+1 prices, return full signal dict.
5. On no match: return skip dict (skipped=True, no trade).

**PatternStrategy** formally extends **BaseStrategy** (ABC).

#### core/trade_manager.py
**Purpose**: Pre-trade gate — now a **passthrough**.

All pre-trade filters (N-2 diff, N-4 win-rate) have been removed. The pattern
strategy handles all entry decisions internally. `TradeManager.check()` always
returns `FilterResult(allowed=True)`. The class is retained so the scheduler
call-site remains intact and can accommodate future filters without changes.

`FilterResult` fields: `allowed`, `reason`, `filter_name`. (Former `n2_side`
and `n4_win` fields removed along with the filters they supported.)

#### core/trader.py
**Purpose**: FOK market order execution with retry logic.

**FOK Order** (Fill-Or-Kill):
- Signs locally via py-clob-client, posts to CLOB
- Amount rounded to 2 decimal places (client workaround)
- Runs in asyncio.to_thread() (library is synchronous)

**Retry Safety** (`place_fok_order_with_retry()`):
1. Time fence: abort if less than 30s until slot end
2. Duplicate guard: check DB before each attempt
3. Fill verification: check CLOB status for MATCHED
4. Max 3 retries with exponential backoff (2s, 4s, cap 5s)
5. Refreshes best ask price between retries

#### core/resolver.py
**Purpose**: Determines trade outcome via Coinbase candle data.

- Fetches 5-min candle for the exact slot window
- Winner: close >= open => "Up", else "Down"
- Retries 5 times at 10s intervals (worst case: 50s)
- Uses wider window [ts-300, ts+600] then filters exact match

#### core/pending_queue.py
**Purpose**: Persistent JSON retry queue for unresolved slots. Survives Railway restarts.

#### core/redeemer.py
**Purpose**: Detects and redeems winning positions on-chain.

**Flow**:
1. Fetch positions from Polymarket Data API
2. Filter: size > 0, market resolved, our outcome is winner
3. Call CTF.redeemPositions() on Polygon via web3.py
4. Contracts: CTF at 0x4D97...6045, USDC.e at 0x2791...4174
5. Gas estimation + 20% buffer, 200k fallback

---

### polymarket/ (API Layer)

#### polymarket/client.py
**Purpose**: Wrapper around py-clob-client. Derives L2 API credentials from private key, re-instantiates with credentials.

#### polymarket/markets.py
**Purpose**: Slot boundaries + price fetching.

- Slot alignment: :00, :05, :10...55 (5-min grid)
- `get_slot_prices(slug)`: Two-step: Gamma API (token IDs) -> CLOB API (best ask prices)
- Uses CLOB best ask (actual buyer price) instead of Gamma mid-price

#### polymarket/account.py
**Purpose**: Account queries: balance (USDC 6 decimals), open positions, connection ping.

---

### db/ (Persistence Layer)

#### db/models.py
**Tables**:
1. **signals**: id, slot_start, slot_end, side, entry_price, outcome, is_win, skipped, filter_blocked
   - `filter_blocked` is retained in schema for backward compatibility but is always 0.
     The pre-trade filter that set it has been removed.
2. **trades**: id, signal_id, side, entry_price, amount_usdc, status, pnl, retry_count, is_demo
3. **settings**: key/value (autotrade_enabled, trade_amount_usdc, demo_trade_enabled, etc.)
4. **redemptions**: condition_id, outcome_index, size, tx_hash, status, gas_used

**Defaults**: autotrade=false, trade_amount=1.0, demo_trade=false, demo_bankroll=1000.00

#### db/queries.py
Comprehensive CRUD + analytics using aiosqlite:
- Settings: get/set, toggle methods
- Signals: insert, resolve, analytics, export
- Trades: insert, update retry, resolve, P&L tracking
- Redemptions: insert, dedup guard, stats
- Demo: separate bankroll management, trade tracking
- Streak computation: current, best win, worst loss

**Removed functions** (were N-2/N-4 filter support, no longer needed):
- `get_n2_trade_side()`, `get_n2_demo_trade_side()`
- `get_n4_trade_win()`, `get_n4_demo_trade_win()`
- `update_signal_filter_blocked()`

---

## Trading Strategy Deep Dive

### Core Strategy: 6-Candle Pattern Matching

The bot reads the last six confirmed-closed 5-min BTC-USD candles from
Coinbase and builds a direction string. If the pattern matches one of the 22
known entries in PATTERN_TABLE, a trade is placed for the predicted N+1
direction.

**Pattern string format** (left to right): [N-1][N-2][N-3][N-4][N-5][N-6]
- N-1 = most recently closed candle
- U = close >= open (bullish), D = close < open (bearish)

**Example**: `DUUUDU` means N-1 down, N-2 up, N-3 up, N-4 up, N-5 down, N-6 up.

**Candle timing safety**: Called at T-85s (215s into a 300s slot), the current
candle is still open. The strategy always drops the most-recent API response
candle before slicing, ensuring only confirmed-closed candles are used.

### Resolution Mechanism

Uses Coinbase BTC-USD candles, NOT Polymarket on-chain resolution. More reliable and deterministic. close >= open = Up, else Down.

---

## Risk Management & Filters

### Position Sizing
- Fixed flat sizing: $1.00 USDC (configurable)
- No martingale, Kelly, or dynamic sizing
- Bankroll clamped to >= 0 in demo mode

### Pre-Trade Filters
All pre-trade filters have been removed. The pattern strategy handles entry
decisions internally via PATTERN_TABLE lookup. `TradeManager` is a passthrough
that always returns `allowed=True`.

### Execution Safeguards
- FOK orders (no partial fills, immediate kill if not fillable)
- 3 retries with backoff, time fence at 30s
- Duplicate guard prevents double-ordering
- Price refresh between retries

---

## P&L Calculation

```
Win:  pnl = amount_usdc * (1/entry_price - 1)
Loss: pnl = -amount_usdc
```

This is the standard Polymarket binary token payout formula.
Buying `amount` at price `p` (where 0 < p < 1) pays out `amount/p` on win.
Profit = amount/p - amount = amount * (1/p - 1).

At $0.51 entry: break-even win rate = ~51%
At $0.60 entry: break-even win rate = ~60%

---

## Data Flow

```
APScheduler (every 5 min)
  -> _check_and_trade()
    -> strategy.check_signal()
         -> PatternStrategy.check_signal()
              -> _fetch_candles() -> Coinbase candles API
              -> _build_pattern_string() -> 6-char pattern
              -> PATTERN_TABLE lookup
              -> markets.get_next_slot_info() + get_slot_prices()
                   -> Gamma API + CLOB API
    -> TradeManager.check() -> always allowed=True (passthrough)
    -> trader.place_fok_order_with_retry() -> py-clob-client
    -> resolver.resolve_slot() -> Coinbase candles
    -> redeemer.scan_and_redeem() -> Data API + web3.py -> Polygon

Telegram
  -> handlers.py -> db.queries -> SQLite
  -> formatters.py (output formatting)
```
