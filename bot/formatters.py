"""Message formatters — every output shows UTC timeslots with emojis."""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Live notifications (sent by scheduler)
# ---------------------------------------------------------------------------

def format_signal(
    side: str,
    entry_price: float,
    slot_start_str: str,
    slot_end_str: str,
    autotrade: bool,
    adx_direction: str | None = None,
    adx_flipped: bool = False,
    adx_value: float | None = None,
    n2_filter_enabled: bool = True,
    n2_side: str | None = None,
) -> str:
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    at_line = "\U0001f916 AutoTrade: ON \u2192 Order Placed" if autotrade else "\U0001f916 AutoTrade: OFF"

    # ADX info line
    adx_line = ""
    if adx_direction is not None and adx_value is not None:
        adx_emoji = "\U0001f4c8" if adx_direction == "rising" else "\U0001f4c9"
        flip_note = " (FLIPPED)" if adx_flipped else ""
        adx_line = f"\u2502 {adx_emoji} ADX(14): {adx_value:.2f} {adx_direction}{flip_note}\n"

    # N-2 filter info line
    n2_line = ""
    if n2_filter_enabled and n2_side is not None:
        n2_emoji = "\U0001f4c8" if n2_side == "Up" else "\U0001f4c9"
        n2_line = f"\u2502 {n2_emoji} N-2 was: {n2_side} \u2713 differs\n"
    elif n2_filter_enabled and n2_side is None:
        n2_line = "\u2502 \u2754 N-2: no prior trade\n"

    return (
        "\U0001f4e1 <b>Signal Fired!</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC\n"
        f"\u2502 {side_emoji} Side: {side}\n"
        f"\u2502 \U0001f4b2 Ask Price: ${entry_price:.2f}\n"
        f"{adx_line}"
        f"{n2_line}"
        f"\u2502 {at_line}\n"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_skip(
    slot_start_str: str,
    slot_end_str: str,
    up_price: float,
    down_price: float,
    adx_direction: str | None = None,
    adx_value: float | None = None,
) -> str:
    adx_line = ""
    if adx_direction is not None and adx_value is not None:
        adx_emoji = "\U0001f4c8" if adx_direction == "rising" else "\U0001f4c9"
        adx_line = f"\u2502 {adx_emoji} ADX(14): {adx_value:.2f} {adx_direction}\n"

    return (
        "\u23ed\ufe0f <b>No Signal</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC\n"
        f"\u2502 \U0001f4c8 Up Ask: ${up_price:.2f}  |  \U0001f4c9 Down Ask: ${down_price:.2f}\n"
        f"{adx_line}"
        "\u2502 Neither side \u2265 $0.51 \u2014 skipping\n"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_filter_blocked(
    side: str,
    slot_start_str: str,
    slot_end_str: str,
    reason: str,
    n2_side: str | None = None,
) -> str:
    """Notification sent when the N-2 filter blocks a trade."""
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    n2_line = ""
    if n2_side:
        n2_emoji = "\U0001f4c8" if n2_side == "Up" else "\U0001f4c9"
        n2_line = f"\u2502 {n2_emoji} N-2 Side: {n2_side}\n"
    return (
        "\U0001f6ab <b>Trade Blocked \u2014 N-2 Filter</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC\n"
        f"\u2502 {side_emoji} Signal: {side}\n"
        f"{n2_line}"
        f"\u2502 \U0001f6ab Reason: {reason}\n"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_resolution(
    is_win: bool,
    side: str,
    entry_price: float,
    slot_start_str: str,
    slot_end_str: str,
    pnl: float | None = None,
) -> str:
    result_price = 1.00 if is_win else 0.00
    icon = "\u2705" if is_win else "\u274c"
    label = "WIN" if is_win else "LOSS"
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"

    lines = [
        f"{icon} <b>Signal Result \u2014 {label}</b>",
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        f"\u2502 \u23f0 Slot: {slot_start_str}-{slot_end_str} UTC",
        f"\u2502 {side_emoji} Side: {side}",
        f"\u2502 \U0001f4b2 Entry: ${entry_price:.2f} \u2192 Result: ${result_price:.2f}",
    ]
    if pnl is not None:
        sign = "+" if pnl >= 0 else ""
        lines.append(f"\u2502 \U0001f4b0 P&L: {sign}${pnl:.2f}")
    lines.append("\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trade execution notifications (sent by scheduler during FOK retry loop)
# ---------------------------------------------------------------------------

def format_trade_filled(
    side: str,
    slot_label: str,
    ask_price: float,
    amount_usdc: float,
    shares: float | None,
    order_id: str | None,
    attempts: int,
) -> str:
    """Rich fill confirmation box sent when a FOK order is MATCHED."""
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    attempt_note = f" (attempt {attempts})" if attempts > 1 else ""
    shares_line = ""
    if shares is not None:
        shares_line = f"\u2502 \U0001f4ca Shares: {shares:.4f}\n"
    oid_line = ""
    if order_id:
        oid_short = (order_id[:10] + "..." + order_id[-6:]) if len(order_id) > 16 else order_id
        oid_line = f"\u2502 \U0001f9fe Order ID: <code>{oid_short}</code>\n"
    return (
        f"\u2705 <b>Trade FILLED{attempt_note}</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 {side_emoji} Side: {side}  \u23f0 {slot_label}\n"
        f"\u2502 \U0001f4b2 Ask Price: ${ask_price:.4f}\n"
        f"\u2502 \U0001f4b5 Amount: ${amount_usdc:.2f} USDC\n"
        f"{shares_line}"
        f"{oid_line}"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_trade_unmatched(
    side: str,
    slot_label: str,
    attempts: int,
    reason: str,
) -> str:
    """Rich failure box sent when all FOK retry attempts are exhausted."""
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    return (
        "\u274c <b>Trade UNMATCHED</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 {side_emoji} Side: {side}  \u23f0 {slot_label}\n"
        f"\u2502 \U0001f504 Attempts: {attempts}\n"
        f"\u2502 \U0001f4cb Reason: {reason}\n"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_trade_aborted(
    side: str,
    slot_label: str,
    reason: str,
) -> str:
    """Rich abort box sent when time fence or duplicate guard fires."""
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    return (
        "\u26d4 <b>Trade ABORTED</b>\n"
        "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u2502 {side_emoji} Side: {side}  \u23f0 {slot_label}\n"
        f"\u2502 \U0001f4cb Reason: {reason}\n"
        "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
    )


def format_trade_retrying(
    side: str,
    slot_label: str,
    attempt: int,
    max_attempts: int,
    reason: str,
) -> str:
    """Compact inline message sent before each retry attempt (no box)."""
    side_emoji = "\U0001f4c8" if side == "Up" else "\U0001f4c9"
    return (
        f"\U0001f504 <b>Trade retrying</b> (attempt {attempt}/{max_attempts}) "
        f"{side_emoji} {side} {slot_label} \u2014 {reason}"
    )


# ---------------------------------------------------------------------------
# Redemption formatters
# ---------------------------------------------------------------------------

def format_redeem_preview(results: list[dict]) -> str:
    """Format /redeem dry-run scan results before user confirms."""
    SEP = "\u2501" * 20
    if not results:
        return (
            "\U0001f4b0 <b>Redeem — No Positions Found</b>\n"
            + SEP + "\n"
            "No redeemable winning positions detected in your wallet.\n"
            "Positions only appear here once the market resolves on-chain."
        )

    lines = [
        f"\U0001f4b0 <b>Redeem Preview ({len(results)} position(s) found)</b>",
        SEP,
    ]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or r.get("condition_id", "Unknown"))[:60]
        size = r.get("size", 0)
        lines.append(f"{i}. {title}")
        lines.append(f"   \U0001f4b0 Size: {size:.4f} shares")
    lines += [
        SEP,
        "Tap <b>Confirm Redeem</b> to execute all redemptions on-chain.",
    ]
    return "\n".join(lines)


def format_redeem_results(results: list[dict]) -> str:
    """Format the outcome after redemption transactions are sent."""
    SEP = "\u2501" * 20
    if not results:
        return (
            "\U0001f4b0 <b>Redeem Complete</b>\n"
            + SEP + "\n"
            "No redeemable positions found — nothing to redeem."
        )

    success_count = sum(1 for r in results if r.get("success"))
    fail_count = len(results) - success_count

    lines = [
        f"\U0001f4b0 <b>Redeem Complete</b>  \u2705 {success_count}  \u274c {fail_count}",
        SEP,
    ]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or r.get("condition_id", "Unknown"))[:55]
        size = r.get("size", 0)
        if r.get("success"):
            tx = r.get("tx_hash", "")
            short_tx = tx[:10] + "..." + tx[-6:] if tx and len(tx) > 16 else (tx or "N/A")
            gas = r.get("gas_used")
            gas_str = f"  gas={gas:,}" if gas else ""
            lines.append(f"\u2705 {i}. {title}")
            lines.append(f"   {size:.4f} shares  tx: <code>{short_tx}</code>{gas_str}")
        else:
            err = (r.get("error") or "unknown error")[:80]
            lines.append(f"\u274c {i}. {title}")
            lines.append(f"   Error: {err}")
    lines.append(SEP)
    return "\n".join(lines)


def format_auto_redeem_notification(results: list[dict]) -> str:
    """Compact notification sent by the auto-redeem scheduler job."""
    success = [r for r in results if r.get("success")]
    failed  = [r for r in results if not r.get("success")]
    SEP = "\u2501" * 20

    lines = [
        f"\U0001f916 <b>Auto-Redeem Complete</b>  \u2705 {len(success)}  \u274c {len(failed)}",
        SEP,
    ]
    for r in success:
        title = (r.get("title") or r.get("condition_id", "?"))[:55]
        tx = r.get("tx_hash", "")
        short_tx = tx[:10] + "..." + tx[-6:] if tx and len(tx) > 16 else (tx or "N/A")
        lines.append(f"\u2705 {title}")
        lines.append(f"   tx: <code>{short_tx}</code>")
    for r in failed:
        title = (r.get("title") or r.get("condition_id", "?"))[:55]
        err = (r.get("error") or "unknown")[:60]
        lines.append(f"\u274c {title}")
        lines.append(f"   {err}")
    lines.append(SEP)
    return "\n".join(lines)


def format_redemption_history(stats: dict, recent: list[dict]) -> str:
    """Format the /redemptions dashboard."""
    SEP = "\u2501" * 20
    lines = [
        "\U0001f4b0 <b>Redemption History</b>",
        SEP,
        f"\U0001f4ca Total Redeemed: {stats['total']}",
        f"\u2705 Success: {stats['success']}  \u274c Failed: {stats['failed']}",
        f"\U0001f4b0 Total Size Redeemed: {stats['total_size']:.4f} shares",
        SEP,
    ]
    if not recent:
        lines.append("No redemptions recorded yet.")
        return "\n".join(lines)

    lines.append("\U0001f4cb <b>Recent Redemptions:</b>")
    for r in recent:
        ts = r.get("created_at", "")[:16]
        title = (r.get("title") or r.get("condition_id", "Unknown"))[:45]
        size = r.get("size", 0)
        status = r.get("status", "?")
        icon = "\u2705" if status == "success" else "\u274c"
        tx = r.get("tx_hash") or ""
        short_tx = tx[:8] + "..." if tx else "N/A"
        lines.append(f"{icon} {ts}  {title}")
        lines.append(f"   {size:.4f} sh  tx: <code>{short_tx}</code>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dashboards (requested via bot commands)
# ---------------------------------------------------------------------------

def format_signal_stats(stats: dict[str, Any], label: str = "All Time") -> str:
    streak_str = "0"
    if stats.get("current_streak") and stats.get("current_streak_type"):
        streak_str = f"{stats['current_streak']}{stats['current_streak_type']}"

    SEP = "\u2501" * 20
    lines = [
        f"\U0001f4ca <b>Signal Performance ({label})</b>",
        SEP,
        f"\U0001f4e1 Total Signals: {stats['total_signals']}",
        f"\u2705 Wins: {stats['wins']}  |  \u274c Losses: {stats['losses']}",
        f"\U0001f4c8 Win Rate: {stats['win_pct']}%",
        SEP,
        f"\U0001f525 Current Streak: {streak_str}",
        f"\U0001f3c6 Best Win Streak: {stats['best_win_streak']}",
        f"\U0001f480 Worst Loss Streak: {stats['worst_loss_streak']}",
        SEP,
        f"\u23ed\ufe0f Skipped (No Signal): {stats['skip_count']}",
        f"\U0001f6ab N-2 Filter Blocked: {stats.get('filter_blocked_count', 0)}",
    ]
    return "\n".join(lines)


def format_trade_stats(stats: dict[str, Any], label: str = "All Time") -> str:
    streak_str = "0"
    if stats.get("current_streak") and stats.get("current_streak_type"):
        streak_str = f"{stats['current_streak']}{stats['current_streak_type']}"

    sign = "+" if stats["net_pnl"] >= 0 else ""
    roi_sign = "+" if stats["roi_pct"] >= 0 else ""

    SEP = "\u2501" * 20
    lines = [
        f"\U0001f4b0 <b>Trade Performance ({label})</b>",
        SEP,
        f"\U0001f4ca Total Trades: {stats['total_trades']}",
        f"\u2705 Wins: {stats['wins']}  |  \u274c Losses: {stats['losses']}",
        f"\U0001f4c8 Win Rate: {stats['win_pct']}%",
        SEP,
        f"\U0001f4b5 Total Deployed: ${stats['total_deployed']:.2f}",
        f"\U0001f4b0 Total Returned: ${stats['total_returned']:.2f}",
        f"\U0001f4c8 Net P&L: {sign}${stats['net_pnl']:.2f}",
        f"\U0001f4ca ROI: {roi_sign}{stats['roi_pct']}%",
        SEP,
        f"\U0001f525 Current Streak: {streak_str}",
        f"\U0001f3c6 Best Win Streak: {stats['best_win_streak']}",
    ]
    return "\n".join(lines)


def format_status(
    connected: bool,
    balance: float | None,
    autotrade: bool,
    trade_amount: float,
    open_positions: int,
    uptime_str: str,
    last_signal: str | None,
    auto_redeem: bool = False,
    n2_filter_enabled: bool = True,
) -> str:
    conn_icon = "\U0001f7e2" if connected else "\U0001f534"
    conn_text = "Connected" if connected else "Disconnected"
    at_text = "ON" if autotrade else "OFF"
    ar_text = "ON" if auto_redeem else "OFF"
    n2_text = "ON" if n2_filter_enabled else "OFF"
    bal_text = f"{balance:.2f} USDC" if balance is not None else "N/A"
    sig_text = last_signal or "None"

    SEP = "\u2501" * 20
    lines = [
        "\U0001f916 <b>AutoPoly Status</b>",
        SEP,
        f"{conn_icon} Bot: Running",
        f"\U0001f517 Polymarket: {conn_text}",
        f"\U0001f4b0 Balance: {bal_text}",
        SEP,
        f"\U0001f916 AutoTrade: {at_text}",
        f"\U0001f9ea N-2 Filter: {n2_text}",
        f"\U0001f4b5 Trade Amount: ${trade_amount:.2f}",
        f"\U0001f4ca Open Positions: {open_positions}",
        f"\U0001f4b0 Auto-Redeem: {ar_text}",
        SEP,
        f"\u23f0 Uptime: {uptime_str}",
        f"\U0001f4e1 Last Signal: {sig_text}",
    ]
    return "\n".join(lines)


def format_recent_signals(signals: list[dict[str, Any]]) -> str:
    if not signals:
        return "\nNo signals recorded yet."
    lines = ["\n\U0001f4cb <b>Recent Signals:</b>"]
    for s in signals:
        ss = s["slot_start"].split(" ")[-1] if " " in s["slot_start"] else s["slot_start"]
        se = s["slot_end"].split(" ")[-1] if " " in s["slot_end"] else s["slot_end"]
        if s["skipped"]:
            lines.append(f"\u23ed\ufe0f {ss}-{se} UTC \u2014 skipped")
        elif s.get("filter_blocked"):
            lines.append(f"\U0001f6ab {ss}-{se} UTC \u2014 N-2 blocked")
        else:
            icon = "\u2705" if s.get("is_win") == 1 else ("\u274c" if s.get("is_win") == 0 else "\u23f3")
            lines.append(f"{icon} {ss}-{se} UTC  {s['side']}  ${s.get('entry_price', 0):.2f}")
    return "\n".join(lines)


def format_recent_trades(trades: list[dict[str, Any]]) -> str:
    if not trades:
        return "\nNo trades recorded yet."
    lines = ["\n\U0001f4cb <b>Recent Trades:</b>"]
    for t in trades:
        ss = t["slot_start"].split(" ")[-1] if " " in t["slot_start"] else t["slot_start"]
        se = t["slot_end"].split(" ")[-1] if " " in t["slot_end"] else t["slot_end"]
        icon = "\u2705" if t.get("is_win") == 1 else ("\u274c" if t.get("is_win") == 0 else "\u23f3")
        pnl_str = ""
        if t.get("pnl") is not None:
            sign = "+" if t["pnl"] >= 0 else ""
            pnl_str = f"  {sign}${t['pnl']:.2f}"
        lines.append(f"{icon} {ss}-{se} UTC  {t['side']}  ${t['amount_usdc']:.2f}{pnl_str}")
    return "\n".join(lines)


def format_help() -> str:
    return (
        "\u2753 <b>AutoPoly Commands</b>\n\n"
        "/start \u2014 Main menu\n"
        "/status \u2014 Bot status & balance\n"
        "/signals \u2014 Signal performance stats\n"
        "/trades \u2014 Trade P&L dashboard\n"
        "/redeem \u2014 Scan & redeem winning positions\n"
        "/redemptions \u2014 Redemption history\n"
        "/settings \u2014 Toggle autotrade/auto-redeem, set amount\n"
        "/help \u2014 This help message\n\n"
        "<b>How it works:</b>\n"
        "Every 5 minutes the bot checks the NEXT slot's BTC up/down "
        "prices 85 seconds before the current slot ends. If either "
        "side \u2265 $0.51, a signal fires and trades that slot.\n\n"
        "<b>ADX Filter:</b>\n"
        "The bot computes ADX(14) from Coinbase BTC-USD 5-minute candles. "
        "If ADX is <b>rising</b> (strengthening trend), the signal is "
        "<b>flipped</b> (Up\u2194Down) to fade the consensus. If ADX is "
        "falling or flat, the original signal is kept. "
        "With AutoTrade ON, a FOK market order is placed automatically.\n\n"
        "<b>N-2 Diff Filter:</b>\n"
        "Trades are only taken when the current signal side DIFFERS from the side "
        "traded two slots ago (N-2). If N-2 had no trade, the filter passes. "
        "Toggle via Settings \u2192 N-2 Filter. Default: ON.\n\n"
        "<b>Auto-Redeem:</b>\n"
        "When enabled, the bot periodically scans your wallet for resolved "
        "winning positions and calls redeemPositions() on the Polygon CTF "
        "contract to collect your USDC.e. Use /redeem for a manual scan."
    )
