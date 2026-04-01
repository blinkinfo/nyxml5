"""Inline keyboard layouts for the Telegram bot."""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\U0001f4ca Status", callback_data="cmd_status"),
            InlineKeyboardButton("\U0001f4e1 Signals", callback_data="cmd_signals"),
        ],
        [
            InlineKeyboardButton("\U0001f4b0 Trades", callback_data="cmd_trades"),
            InlineKeyboardButton("\u2699\ufe0f Settings", callback_data="cmd_settings"),
        ],
        [
            InlineKeyboardButton("\U0001f4b0 Redeem", callback_data="cmd_redeem"),
            InlineKeyboardButton("\U0001f4dc Redemptions", callback_data="cmd_redemptions"),
        ],
        [
            InlineKeyboardButton("\u2753 Help", callback_data="cmd_help"),
        ],
    ])


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def settings_keyboard(
    autotrade_on: bool,
    trade_amount: float,
    auto_redeem_on: bool = False,
    n2_filter_on: bool = True,
) -> InlineKeyboardMarkup:
    at_label = "\U0001f916 AutoTrade: ON" if autotrade_on else "\U0001f916 AutoTrade: OFF"
    n2_label = "\U0001f9ea N-2 Filter: ON" if n2_filter_on else "\U0001f9ea N-2 Filter: OFF"
    ar_label = "\U0001f4b0 Auto-Redeem: ON" if auto_redeem_on else "\U0001f4b0 Auto-Redeem: OFF"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(at_label, callback_data="toggle_autotrade")],
        [InlineKeyboardButton(n2_label, callback_data="toggle_n2_filter")],
        [InlineKeyboardButton(f"\U0001f4b5 Trade Amount: ${trade_amount:.2f}", callback_data="change_amount")],
        [InlineKeyboardButton(ar_label, callback_data="toggle_auto_redeem")],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Filter rows (Last 10 / Last 50 / All Time)
# ---------------------------------------------------------------------------

def signal_filter_row(active: str = "all") -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            ("[Last 10]" if active == "10" else "Last 10"),
            callback_data="signals_10",
        ),
        InlineKeyboardButton(
            ("[Last 50]" if active == "50" else "Last 50"),
            callback_data="signals_50",
        ),
        InlineKeyboardButton(
            ("[All Time]" if active == "all" else "All Time"),
            callback_data="signals_all",
        ),
    ]
    return InlineKeyboardMarkup([
        buttons,
        [
            InlineKeyboardButton("\U0001f4e5 Download CSV", callback_data="download_csv"),
            InlineKeyboardButton("\U0001f4e5 Download Excel", callback_data="download_xlsx"),
        ],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


def trade_filter_row(active: str = "all") -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            ("[Last 10]" if active == "10" else "Last 10"),
            callback_data="trades_10",
        ),
        InlineKeyboardButton(
            ("[Last 50]" if active == "50" else "Last 50"),
            callback_data="trades_50",
        ),
        InlineKeyboardButton(
            ("[All Time]" if active == "all" else "All Time"),
            callback_data="trades_all",
        ),
    ]
    return InlineKeyboardMarkup([
        buttons,
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Back button only
# ---------------------------------------------------------------------------

def back_to_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Download keyboard (standalone — kept for direct use if needed)
# ---------------------------------------------------------------------------

def download_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\U0001f4e5 Download CSV", callback_data="download_csv"),
            InlineKeyboardButton("\U0001f4e5 Download Excel", callback_data="download_xlsx"),
        ],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


# ---------------------------------------------------------------------------
# Redeem keyboards
# ---------------------------------------------------------------------------

def redeem_confirm_keyboard() -> InlineKeyboardMarkup:
    """Shown after a dry-run scan — lets user confirm or cancel."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\u2705 Confirm Redeem", callback_data="redeem_confirm"),
            InlineKeyboardButton("\u274c Cancel",          callback_data="redeem_cancel"),
        ],
        [InlineKeyboardButton("\U0001f519 Back to Menu", callback_data="cmd_menu")],
    ])


def redeem_done_keyboard() -> InlineKeyboardMarkup:
    """Shown after redemptions complete."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("\U0001f4dc History", callback_data="cmd_redemptions"),
            InlineKeyboardButton("\U0001f519 Menu",    callback_data="cmd_menu"),
        ],
    ])
