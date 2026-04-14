"""Model evaluator — full hold-out test evaluation with metrics table."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from config import ML_PAYOUT_RATIO

log = logging.getLogger(__name__)


def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    test_period_days: float = 37,
    payout: float = ML_PAYOUT_RATIO,
) -> dict:
    """Full evaluation of a trained LightGBM model on a hold-out test set.

    Prints a clear summary table and returns a metrics dict.

    Args:
        model: lgb.Booster instance
        X_test: Feature matrix (n_samples, 22)
        y_test: True binary labels
        threshold: Decision threshold from val-set sweep
        test_period_days: How many days the test set covers (for trades/day)
        payout: Profit per $1 wagered on a winning trade (default:
                ML_PAYOUT_RATIO from config, overridable via ML_PAYOUT_RATIO
                env var). Used to compute payout-adjusted EV/day.

    Returns:
        dict with: wr, precision, recall, f1, trades, trades_per_day,
                   ev_per_trade, ev_per_day, brier_score, calibration_mean,
                   confusion_matrix, threshold, payout
    """
    probs = model.predict(X_test)
    mask = probs >= threshold
    trades = int(mask.sum())

    if trades == 0:
        result = {
            "wr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "trades": 0,
            "trades_per_day": 0.0,
            "ev_per_trade": 0.0,
            "ev_per_day": 0.0,
            "brier_score": float(np.mean((probs - y_test) ** 2)),
            "calibration_mean": float(np.mean(probs)),
            "confusion_matrix": [[0, 0], [0, 0]],
            "threshold": threshold,
            "payout": payout,
        }
        _print_table(result)
        return result

    y_pred = mask.astype(int)
    y_sel = y_test[mask]

    wr = float(y_sel.mean())
    trades_per_day = trades / test_period_days if test_period_days > 0 else 0.0

    # Payout-adjusted EV: accounts for asymmetric win/loss payouts.
    # EV/trade = (WR * payout) - ((1 - WR) * 1.0) = WR * (1 + payout) - 1.0
    ev_per_trade = float(wr * (1.0 + payout) - 1.0)
    ev_per_day = ev_per_trade * trades_per_day

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Brier score (calibration quality)
    brier = float(np.mean((probs - y_test) ** 2))
    calib_mean = float(np.mean(probs[mask]))

    cm = confusion_matrix(y_test, y_pred).tolist()

    result = {
        "wr": wr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trades": trades,
        "trades_per_day": trades_per_day,
        "ev_per_trade": ev_per_trade,
        "ev_per_day": ev_per_day,
        "brier_score": brier,
        "calibration_mean": calib_mean,
        "confusion_matrix": cm,
        "threshold": threshold,
        "payout": payout,
    }

    _print_table(result)
    return result


def compute_risk_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    payout: float,
) -> dict:
    """Compute risk/drawdown metrics for a model at a given threshold.

    Simulates a $1 flat-bet equity curve on the ordered sequence of trades
    selected by the model (probs >= threshold), preserving time order.

    On a $1 flat-bet:
        WIN  -> +payout   (e.g. +$0.85)
        LOSS -> -1.00

    Metrics returned
    ----------------
    max_dd_dollar   : float  — worst peak-to-trough drawdown in dollars
    max_dd_pct      : float  — worst drawdown as % of peak equity
    max_loss_streak : int    — longest consecutive losing trades
    max_win_streak  : int    — longest consecutive winning trades
    profit_factor   : float  — gross wins / gross losses (inf if no losses)
    sharpe          : float  — annualised Sharpe ratio (252 trading days,
                               assuming trades_per_day from this sample)
    trades          : int    — number of trades at this threshold

    All values are 0.0 / 0 when there are no trades at the threshold.
    """
    mask = probs >= threshold
    trades = int(mask.sum())

    _zero: dict = {
        "max_dd_dollar": 0.0,
        "max_dd_pct": 0.0,
        "max_loss_streak": 0,
        "max_win_streak": 0,
        "profit_factor": 0.0,
        "sharpe": 0.0,
        "trades": 0,
    }
    if trades == 0:
        return _zero

    # Ordered W/L outcomes — time order preserved (no shuffling)
    outcomes = y_true[mask].astype(int)  # 1 = win, 0 = loss

    # Per-trade P&L: win -> +payout, loss -> -1.0
    pnl = np.where(outcomes == 1, payout, -1.0)

    # -----------------------------------------------------------------------
    # Max drawdown — peak-to-trough on cumulative equity curve
    # Starting equity = 0 (relative, $1 flat-bet each trade)
    # -----------------------------------------------------------------------
    equity = np.concatenate([[0.0], np.cumsum(pnl)])   # shape (trades+1,)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak                            # always <= 0
    max_dd_dollar = float(np.min(drawdown))            # most negative value

    # Percentage drawdown: drawdown / peak — guard against peak == 0.
    #
    # Design choice: when the running peak is 0 (i.e. equity has never risen
    # above the starting value of 0), we define the percentage drawdown as 0.0
    # at that point rather than divide-by-zero.  This arises in two cases:
    #   1. The very first trade(s) are losses — peak stays at 0.0 while equity
    #      goes negative, so there is no positive capital base to express the
    #      drawdown against.
    #   2. An all-losing sequence — peak never exceeds 0.0 throughout.
    #
    # Consequence: on sequences that open with losses, max_dd_pct will be
    # 0.0% even though max_dd_dollar is negative.  The dollar figure is the
    # authoritative risk measure in that scenario; the percentage figure is
    # intentionally suppressed to avoid a misleading division artefact.
    with np.errstate(invalid="ignore", divide="ignore"):
        dd_pct = np.where(peak > 0, drawdown / peak * 100.0, 0.0)
    max_dd_pct = float(np.min(dd_pct))                 # most negative %

    # -----------------------------------------------------------------------
    # Win / loss streaks
    # -----------------------------------------------------------------------
    max_win_streak = 0
    max_loss_streak = 0
    cur_win = 0
    cur_loss = 0
    for o in outcomes:
        if o == 1:
            cur_win += 1
            cur_loss = 0
            if cur_win > max_win_streak:
                max_win_streak = cur_win
        else:
            cur_loss += 1
            cur_win = 0
            if cur_loss > max_loss_streak:
                max_loss_streak = cur_loss

    # -----------------------------------------------------------------------
    # Profit factor — gross_wins / gross_losses
    # -----------------------------------------------------------------------
    gross_wins   = float(np.sum(pnl[pnl > 0]))
    gross_losses = float(np.abs(np.sum(pnl[pnl < 0])))
    if gross_losses == 0.0:
        profit_factor = float("inf") if gross_wins > 0 else 0.0
    else:
        profit_factor = round(gross_wins / gross_losses, 4)

    # -----------------------------------------------------------------------
    # Sharpe ratio — annualised, assuming ~288 5-min slots/day on this sample
    # We use per-trade returns (pnl) and scale to daily assuming trades_per_day.
    # Formula: (mean_pnl / std_pnl) * sqrt(trades) is the per-sample Sharpe.
    # We annualise by scaling to 252 trading days, so we need trades_per_day.
    # Approximation: trades / (len(probs) * 5 / 1440) gives trades_per_day.
    # -----------------------------------------------------------------------
    sharpe = 0.0
    if trades >= 2:
        mean_r = float(np.mean(pnl))
        std_r  = float(np.std(pnl, ddof=1))
        if std_r > 0:
            # trades_per_day from this sample window
            tpd = trades / max(len(probs) * 5 / 1440, 1e-9)
            # Annualise: multiply by sqrt(252 * trades_per_day)
            sharpe = round((mean_r / std_r) * (252 * tpd) ** 0.5, 4)

    return {
        "max_dd_dollar": round(max_dd_dollar, 4),
        "max_dd_pct": round(max_dd_pct, 4),
        "max_loss_streak": int(max_loss_streak),
        "max_win_streak": int(max_win_streak),
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "trades": trades,
    }


def _print_table(m: dict) -> None:
    """Print a readable evaluation summary."""
    print("\n" + "=" * 52)
    print("  MODEL EVALUATION (HOLD-OUT TEST SET)")
    print("=" * 52)
    print(f"  Threshold          : {m['threshold']:.3f}")
    print(f"  Payout ratio       : {m.get('payout', ML_PAYOUT_RATIO):.2f}")
    print(f"  Win Rate (WR)      : {m['wr']:.4f}  ({m['wr']*100:.2f}%)")
    print(f"  Precision          : {m['precision']:.4f}")
    print(f"  Recall             : {m['recall']:.4f}")
    print(f"  F1                 : {m['f1']:.4f}")
    print(f"  Trades total       : {m['trades']}")
    print(f"  Trades / day       : {m['trades_per_day']:.2f}")
    print(f"  EV / trade ($1)    : {m.get('ev_per_trade', 0.0):+.4f}")
    print(f"  EV / day ($1 flat) : {m.get('ev_per_day', 0.0):+.4f}")
    print(f"  Brier score        : {m['brier_score']:.4f}")
    print(f"  Mean prob (trades) : {m['calibration_mean']:.4f}")
    if m.get("confusion_matrix"):
        cm = m["confusion_matrix"]
        if len(cm) == 2 and len(cm[0]) == 2:
            print("  Confusion matrix   :")
            print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
            print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print("=" * 52 + "\n")
