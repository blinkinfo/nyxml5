"""Trade Manager — pre-trade gate (now a passthrough).

All pre-trade filters (N-2 diff, N-4 win) have been removed.
The new pattern-based strategy handles all entry logic internally.
TradeManager always returns allowed=True for backward compatibility
with the scheduler flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterResult:
    """Result returned by TradeManager.check().

    Fields
    ------
    allowed     : Always True in the current passthrough implementation.
    reason      : Human-readable explanation (logged by scheduler).
    filter_name : Name of the filter that blocked the trade, or None if
                  allowed.  Always None in the passthrough.

    Note: The former n2_side and n4_win fields have been removed.  Those
    fields belonged to the old N-2 direction-diff and N-4 win-rate filters
    which no longer exist.  The pattern strategy handles all entry decisions
    internally and TradeManager is now a transparent passthrough.
    """
    allowed:     bool
    reason:      str
    filter_name: str | None = None


class TradeManager:
    """Stateless pre-trade gate — now always passes.

    Kept as a class so the scheduler import chain stays intact.
    The pattern strategy handles all entry decisions internally.
    """

    @classmethod
    async def check(
        cls,
        signal_side: str,
        current_slot_ts: int,
        is_demo: bool = False,
    ) -> FilterResult:
        """Always return allowed=True.

        All entry filters have been removed in favour of pattern-based
        strategy logic.  This method is kept so the scheduler can call it
        without branching, and so the call-site contract is preserved for
        any future filter that may be re-introduced.
        """
        return FilterResult(
            allowed=True,
            reason="No filters active — passthrough",
        )
