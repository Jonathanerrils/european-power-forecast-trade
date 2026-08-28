"""Ex-post oracle diagnostic (docs/economic_contract_v1.md) -- computes
the best POSSIBLE single-cycle outcome for a delivery day using REALIZED
prices, i.e. with perfect hindsight. This is NEVER an executable
strategy: it cannot be computed D-1, since it requires knowing prices
that haven't cleared yet.

ARCHITECTURAL ISOLATION, not just a naming convention: src/strategy.py
must never import this module. tests/test_strategy_architecture.py
enforces this with a static check, so a future edit that tried to wire
the oracle into live decision-making would fail a test immediately,
not silently ship. Every value computed here must be reported labeled
"ex-post oracle diagnostic, not an executable strategy" -- never bare
"performance."
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.strategy import candidate_pairs, degradation_cost


def best_realized_pair(actual_prices: np.ndarray, eta_rt: float, C: float) -> Tuple[Optional[int], Optional[int], float]:
    """(i*, j*) = argmax over i<j of [eta_rt * P_actual_j - P_actual_i - C],
    using REALIZED prices -- this is hindsight, not a forecast-based
    decision. Floored at 0 (no-trade): a perfect-foresight agent would
    never be forced into a loss, since not trading is always available
    to it too, exactly like every executable strategy.
    """
    n = len(actual_prices)
    best_score = 0.0
    best_pair = (None, None)
    for i, j in candidate_pairs(n):
        score = eta_rt * actual_prices[j] - actual_prices[i] - C
        if score > best_score:
            best_score = score
            best_pair = (i, j)
    return best_pair[0], best_pair[1], best_score


def oracle_pnl(actual_prices: np.ndarray, eta_rt: float, c: float) -> float:
    """Pi_D_oracle = max(0, max over i<j of [eta_rt * P_actual_j - P_actual_i - C]).
    The max(0, ...) floor is enforced by best_realized_pair's own
    best_score initialization at 0.0 -- there is no separate clamp
    step to forget.
    """
    C = degradation_cost(eta_rt, c)
    _, _, best_score = best_realized_pair(actual_prices, eta_rt, C)
    return best_score


def value_capture_ratio(strategy_pnls: np.ndarray, oracle_pnls: np.ndarray) -> Optional[float]:
    """VCR = sum(strategy_pnls) / sum(oracle_pnls), over the SAME set
    of days for both arrays (the caller's responsibility to have
    already restricted both to common_strategy_evaluation_days -- this
    function does not know what a "day" is, only that positions must
    align). Returns None (not NaN, not raising) when the oracle sum is
    <= 0, since VCR is undefined/not meaningful in that case -- an
    oracle that captured nothing has nothing for any strategy to be a
    fraction of.
    """
    if len(strategy_pnls) != len(oracle_pnls):
        raise ValueError("strategy_pnls and oracle_pnls must be the same length (same days, aligned).")
    oracle_sum = float(np.sum(oracle_pnls))
    if oracle_sum <= 0:
        return None
    return float(np.sum(strategy_pnls)) / oracle_sum
