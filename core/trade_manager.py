"""Trade Manager — pre-trade gate with pluggable filters.

Currently implements the 'Diff Side from N-2' filter:
  - Only take a trade if the current signal side DIFFERS from the side
    traded at slot N-2 (two slots ago).
  - If N-2 had no trade (skipped, filtered, or bot was offline), the
    filter passes (we allow the trade — no data = no block).
  - Toggle stored in DB settings key 'n2_filter_enabled' (default: true).

Calling convention:
    result = await TradeManager.check(signal_side, current_slot_ts)
    if result.allowed:
        # proceed with trade
    else:
        # log result.reason, record block in DB, notify Telegram
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from db import queries

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterResult:
    allowed: bool
    reason: str
    filter_name: str | None = None
    n2_side: str | None = None   # what N-2 traded (for logging/notification)


class TradeManager:
    """Stateless pre-trade gate.  All methods are async class methods."""

    @classmethod
    async def check(cls, signal_side: str, current_slot_ts: int) -> FilterResult:
        """Run all active filters. Returns FilterResult(allowed=True/False).

        Filters run in order. First block wins.
        """
        # --- Filter 1: Diff Side from N-2 ---
        n2_result = await cls._check_n2_filter(signal_side, current_slot_ts)
        if not n2_result.allowed:
            return n2_result

        # All filters passed
        return FilterResult(allowed=True, reason="All filters passed")

    # ------------------------------------------------------------------
    # Filter implementations
    # ------------------------------------------------------------------

    @classmethod
    async def _check_n2_filter(cls, signal_side: str, current_slot_ts: int) -> FilterResult:
        """Diff Side from N-2 filter.

        Block if: filter enabled AND N-2 trade side == current signal side.
        Pass if:  filter disabled OR N-2 had no trade OR sides differ.
        """
        enabled = await queries.is_n2_filter_enabled()
        if not enabled:
            return FilterResult(
                allowed=True,
                reason="N-2 filter disabled",
                filter_name="n2_diff",
            )

        n2_side = await queries.get_n2_trade_side(current_slot_ts)

        if n2_side is None:
            # No N-2 trade data — allow (bot was offline, slot was skipped, etc.)
            log.debug(
                "N-2 filter: no trade found for N-2 slot (ts=%d) — allowing %s",
                current_slot_ts - 600,  # approximate, real value computed in query
                signal_side,
            )
            return FilterResult(
                allowed=True,
                reason="N-2 has no trade — filter passes by default",
                filter_name="n2_diff",
                n2_side=None,
            )

        if n2_side == signal_side:
            log.info(
                "N-2 filter BLOCKED: current=%s matches N-2=%s — skipping trade",
                signal_side,
                n2_side,
            )
            return FilterResult(
                allowed=False,
                reason=f"N-2 traded {n2_side} — same as current signal ({signal_side})",
                filter_name="n2_diff",
                n2_side=n2_side,
            )

        log.info(
            "N-2 filter PASSED: current=%s differs from N-2=%s — allowing trade",
            signal_side,
            n2_side,
        )
        return FilterResult(
            allowed=True,
            reason=f"N-2 traded {n2_side} — differs from current ({signal_side})",
            filter_name="n2_diff",
            n2_side=n2_side,
        )
