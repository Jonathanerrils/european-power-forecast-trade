"""Pure economic-kernel functions for the frozen stylised single-cycle
DE-LU day-ahead storage strategy (docs/economic_contract_v1.md).

This module knows NOTHING about: XGBoost fitting, cross-validation,
2026, plotting, or aggregate reporting. It operates on one delivery
day's already-prepared arrays (point forecasts, price bounds, actual
prices, all in chronological order for that day's actual delivery
intervals -- 23/24/25 as they occur) and returns a decision and a P&L.
Day-level orchestration (which day's data to prepare, how to iterate
across the development period) belongs in a separate script, not here.

CRITICAL ARCHITECTURAL RULE, enforced by tests/test_strategy_architecture.py:
this module must NEVER import src/oracle.py. The oracle uses realized
prices to choose a decision; if strategy-generation code could reach
it, that would be look-ahead by construction, not by accident. Keeping
them in separate modules with a one-directional import boundary makes
this a structural guarantee, not a convention someone has to remember.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def assert_no_holdout_access(timestamps, holdout_start_utc=None) -> None:
    """Raises if any timestamp is >= the 2026 holdout boundary. Every
    day-level strategy iteration must call this before doing anything
    else with a batch of timestamps -- the same holdout-protection
    discipline already required elsewhere in this project (README
    "Freeze & holdout protocol"), applied at the strategy layer's own
    entry point rather than assumed inherited from upstream code.
    """
    import pandas as pd
    from src.clean import local_delivery_date_to_utc

    if holdout_start_utc is None:
        holdout_start_utc = local_delivery_date_to_utc("2026-01-01")
    ts = pd.to_datetime(pd.Series(list(timestamps)), utc=True)
    violations = ts[ts >= holdout_start_utc]
    if len(violations) > 0:
        raise ValueError(
            f"{len(violations)} timestamp(s) at or after the 2026 holdout boundary "
            f"({holdout_start_utc}) -- strategy code must never read holdout data. "
            f"Earliest violation: {violations.min()}."
        )


def degradation_cost(eta_rt: float, c: float) -> float:
    """C = c * (1 + eta_rt) -- cost per MWh of THROUGHPUT (charge +
    discharge summed), not a flat per-cycle fee. See economic_contract_v1.md
    "Costs" section for why this distinction is large enough to flip
    trade/no-trade decisions (85% higher at eta=0.85 than a flat-c
    interpretation would give).
    """
    if not (0 < eta_rt <= 1):
        raise ValueError(f"eta_rt must be in (0, 1], got {eta_rt}")
    if c < 0:
        raise ValueError(f"c must be non-negative, got {c}")
    return c * (1 + eta_rt)


def price_bounds(point_forecast: pd.Series, q10_offset: pd.Series, q90_offset: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Returns (L, U) = (point_forecast + q10_offset, point_forecast + q90_offset)
    -- ACTUAL PRICE BOUNDS, not raw residual-quantile offsets. This
    function exists specifically so nothing downstream can accidentally
    use a raw offset (a small number, e.g. +/-5 to 30 EUR/MWh) where an
    absolute price (e.g. ~80 EUR/MWh) is required -- see
    economic_contract_v1.md's explicit warning about this exact failure
    mode, and tests/test_strategy.py::test_price_bounds_add_offset_to_forecast.

    Requires all three Series to share an IDENTICAL index -- ordinary
    pandas Series addition silently aligns by label, not position, so
    a length match alone does not guarantee the offsets are attached
    to the correct hours. A silent misalignment here would attach the
    wrong uncertainty bound to the wrong hour without ever raising.
    """
    if not (point_forecast.index.equals(q10_offset.index) and point_forecast.index.equals(q90_offset.index)):
        raise ValueError(
            "point_forecast, q10_offset, and q90_offset must share an identical index -- "
            "pandas Series addition aligns by label, not position, so mismatched indexes "
            "would silently attach offsets to the wrong hours rather than raising."
        )
    L = point_forecast + q10_offset
    U = point_forecast + q90_offset
    return L, U


def _validate_finite_equal_length(arrays: dict) -> None:
    """Fail-closed input validation shared by every decision-making
    entry point. Rejects: (a) arrays of different lengths -- silently
    proceeding would let the optimizer choose a pair using one array's
    indexing while realized_pnl() prices it against a DIFFERENT array
    representing different hours, producing a plausible-looking but
    meaningless result; (b) any NaN/Inf value -- NaN comparisons always
    evaluate False in Python, so an un-validated NaN would cause the
    optimizer to silently SKIP that candidate rather than fail, exactly
    the kind of "quietly ignore a bad hour" behavior this project has
    refused everywhere else.
    """
    lengths = {name: len(arr) for name, arr in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"All arrays must have the same length, got: {lengths}")
    for name, arr in arrays.items():
        arr_np = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(arr_np)):
            n_bad = int((~np.isfinite(arr_np)).sum())
            raise ValueError(f"'{name}' contains {n_bad} non-finite value(s) (NaN/Inf) -- refusing to silently skip them.")


def candidate_pairs(n_hours: int) -> List[Tuple[int, int]]:
    """All valid (i, j) index pairs with i < j for a day with n_hours
    chronologically-ordered delivery intervals (23, 24, or 25 -- DST
    days handled by construction, since this only needs n, not a
    hardcoded 24).
    """
    if n_hours < 2:
        return []
    return [(i, j) for i in range(n_hours) for j in range(i + 1, n_hours)]


def best_point_pair(point_forecast: np.ndarray, eta_rt: float, C: float) -> Tuple[Optional[int], Optional[int], float]:
    """Point-forecast decision rule: (i*, j*) = argmax over i<j of
    [eta_rt * P_hat_j - P_hat_i - C]. Returns (None, None, 0.0) --
    no_trade -- if the best achievable score is <= 0. Never forces a
    trade the forecast itself predicts will lose money.
    """
    _validate_finite_equal_length({"point_forecast": point_forecast})
    n = len(point_forecast)
    best_score = 0.0
    best_pair = (None, None)
    for i, j in candidate_pairs(n):
        score = eta_rt * point_forecast[j] - point_forecast[i] - C
        if score > best_score:
            best_score = score
            best_pair = (i, j)
    return best_pair[0], best_pair[1], (best_score if best_pair[0] is not None else 0.0)


def best_uncertainty_pair(L: np.ndarray, U: np.ndarray, eta_rt: float, C: float) -> Tuple[Optional[int], Optional[int], float]:
    """Uncertainty-aware decision rule: (i*, j*) = argmax over i<j of
    [eta_rt * L_j - U_i - C], where L/U are ALREADY price bounds
    (point forecast + residual offset -- see price_bounds()), not raw
    offsets. Same no-forced-trade floor as best_point_pair.

    This is a conservative DECISION SCORE, not a jointly-calibrated
    confidence interval for the inter-hour spread -- L and U are
    marginal per-hour bounds, and buy/sell-hour forecast errors are
    plausibly correlated. See economic_contract_v1.md for why this is
    an intentional, documented simplification, not an oversight.
    """
    _validate_finite_equal_length({"L": L, "U": U})
    n = len(L)
    best_score = 0.0
    best_pair = (None, None)
    for i, j in candidate_pairs(n):
        score = eta_rt * L[j] - U[i] - C
        if score > best_score:
            best_score = score
            best_pair = (i, j)
    return best_pair[0], best_pair[1], (best_score if best_pair[0] is not None else 0.0)


def realized_pnl(actual_prices: np.ndarray, i: Optional[int], j: Optional[int], eta_rt: float, C: float) -> Tuple[float, float]:
    """Returns (gross_pnl, net_pnl) for a given (possibly None, None =
    no_trade) decision, evaluated against REALIZED prices. This
    function is the only place forecast-vs-actual distinction matters:
    the decision (i, j) must already be fixed before this is called --
    this function never chooses a pair, it only prices one.
    """
    if i is None or j is None:
        return 0.0, 0.0
    if not (i < j):
        raise AssertionError(f"i must be < j, got i={i}, j={j} -- a battery cannot discharge before it charges")
    gross = eta_rt * actual_prices[j] - actual_prices[i]
    net = gross - C
    return gross, net


def run_day(
    point_forecast: np.ndarray,
    actual_prices: np.ndarray,
    eta_rt: float,
    c: float,
    L: np.ndarray = None,
    U: np.ndarray = None,
) -> dict:
    """Orchestrates one delivery day's decision + realized P&L for
    EITHER the point-forecast rule (L/U omitted) or the
    uncertainty-aware rule (L/U provided) -- never both at once for a
    single call, matching the contract's S2/S4 (point) vs S3/S5
    (uncertainty-aware) split. Returns a dict with the decision and
    both gross/net realized P&L.

    Validates that EVERY provided array (point_forecast, actual_prices,
    and L/U if given) has the SAME length before doing anything else --
    a length mismatch here would otherwise let the decision be chosen
    from one set of hours and priced against a DIFFERENT set, which
    can silently produce a plausible-looking but meaningless result
    whenever the chosen pair happens to fall within the shorter
    array's bounds (it does not always crash).
    """
    arrays_to_check = {"point_forecast": point_forecast, "actual_prices": actual_prices}
    if L is not None:
        arrays_to_check["L"] = L
    if U is not None:
        arrays_to_check["U"] = U
    _validate_finite_equal_length(arrays_to_check)

    C = degradation_cost(eta_rt, c)
    if L is not None or U is not None:
        if L is None or U is None:
            raise ValueError("Both L and U must be provided together, or neither.")
        i, j, decision_score = best_uncertainty_pair(L, U, eta_rt, C)
    else:
        i, j, decision_score = best_point_pair(point_forecast, eta_rt, C)

    gross, net = realized_pnl(actual_prices, i, j, eta_rt, C)
    return {
        "i": i, "j": j, "traded": i is not None,
        "decision_score": decision_score,
        "gross_pnl": gross, "net_pnl": net,
    }
