"""Tests for src/oracle.py -- the ex-post diagnostic. Test 6 (oracle
floor) and test 8 (oracle dominance) from docs/economic_contract_v1.md
live here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.oracle import best_realized_pair, oracle_pnl, value_capture_ratio
from src.strategy import best_point_pair, realized_pnl, degradation_cost


def test_oracle_finds_the_obviously_best_pair():
    actual = np.array([50.0, 30.0, 20.0, 40.0, 70.0, 100.0])
    i, j, score = best_realized_pair(actual, eta_rt=0.85, C=5.0)
    assert (i, j) == (2, 5)
    assert score == pytest.approx(0.85 * 100.0 - 20.0 - 5.0)


def test_oracle_floor_never_negative():
    """Test 6: a day where EVERY (i, j) pair is unprofitable under
    realized prices must produce Pi_oracle = 0, never a negative value
    -- a perfect-foresight agent would simply choose not to trade.
    """
    actual = np.array([50.0, 50.5, 50.2, 50.8, 50.1])  # flat, tiny spreads
    score = oracle_pnl(actual, eta_rt=0.85, c=20.0)  # cost swamps any spread here
    assert score == 0.0
    assert score >= 0.0


def test_oracle_dominance_holds_across_many_randomized_scenarios():
    """Test 8: for every strategy and every day, strategy P&L must
    never exceed the oracle's -- the oracle searches every valid pair
    (plus no-trade) using the SAME realized prices the strategy's own
    P&L is computed from. This committed test runs 2,000 randomized
    trials, including deliberately suboptimal strategy choices; a
    separate, larger 20,000-trial exploratory check was run once
    before writing docs/economic_contract_v1.md's dominance claim, but
    2,000 plus the underlying deterministic reasoning (the oracle is
    a strict superset search over the same realized prices) is ample
    for an ongoing regression test -- no need to pay the runtime cost
    of 20,000 on every test run for the same statistical confidence.
    """
    rng = np.random.default_rng(0)
    eta_rt, c = 0.85, 10.0
    C = degradation_cost(eta_rt, c)
    violations = 0
    for _ in range(2000):
        n = rng.choice([23, 24, 25])
        actual = rng.uniform(-50, 300, n)
        oracle_score = oracle_pnl(actual, eta_rt, c)

        # A strategy computed from a DELIBERATELY WRONG forecast (pure
        # noise, unrelated to actual) -- likely picks a suboptimal pair.
        bad_forecast = rng.uniform(-50, 300, n)
        i, j, _ = best_point_pair(bad_forecast, eta_rt, C)
        _, strategy_net = realized_pnl(actual, i, j, eta_rt, C)

        if strategy_net > oracle_score + 1e-9:
            violations += 1
    assert violations == 0


def test_oracle_dominance_even_when_strategy_uses_correct_forecast():
    """The dominance property must hold even for a 'good' strategy
    (one whose forecast matches actual prices closely) -- the oracle
    can only be at least as good, since it has strictly more
    information (perfect hindsight vs. a forecast, however accurate).
    """
    rng = np.random.default_rng(1)
    eta_rt, c = 0.85, 10.0
    C = degradation_cost(eta_rt, c)
    for _ in range(500):
        n = rng.choice([23, 24, 25])
        actual = rng.uniform(-50, 300, n)
        forecast = actual + rng.normal(0, 5, n)  # close but imperfect
        oracle_score = oracle_pnl(actual, eta_rt, c)
        i, j, _ = best_point_pair(forecast, eta_rt, C)
        _, strategy_net = realized_pnl(actual, i, j, eta_rt, C)
        assert strategy_net <= oracle_score + 1e-9


def test_value_capture_ratio_computes_correctly():
    strategy_pnls = np.array([10.0, 20.0, 0.0])
    oracle_pnls = np.array([15.0, 25.0, 5.0])
    vcr = value_capture_ratio(strategy_pnls, oracle_pnls)
    assert vcr == pytest.approx(30.0 / 45.0)


def test_value_capture_ratio_never_exceeds_one_given_dominance():
    """Corollary of oracle dominance: if strategy_pnls[d] <= oracle_pnls[d]
    for every day, the ratio of sums cannot exceed 1.
    """
    rng = np.random.default_rng(2)
    strategy_pnls, oracle_pnls = [], []
    eta_rt, c = 0.85, 10.0
    C = degradation_cost(eta_rt, c)
    for _ in range(200):
        n = rng.choice([23, 24, 25])
        actual = rng.uniform(-50, 300, n)
        forecast = actual + rng.normal(0, 5, n)
        oracle_pnls.append(oracle_pnl(actual, eta_rt, c))
        i, j, _ = best_point_pair(forecast, eta_rt, C)
        _, net = realized_pnl(actual, i, j, eta_rt, C)
        strategy_pnls.append(net)
    vcr = value_capture_ratio(np.array(strategy_pnls), np.array(oracle_pnls))
    assert vcr is not None
    assert vcr <= 1.0 + 1e-9


def test_value_capture_ratio_returns_none_for_nonpositive_oracle_sum():
    strategy_pnls = np.array([0.0, 0.0])
    oracle_pnls = np.array([0.0, 0.0])
    assert value_capture_ratio(strategy_pnls, oracle_pnls) is None


def test_value_capture_ratio_can_be_negative_if_strategy_loses():
    strategy_pnls = np.array([-10.0])
    oracle_pnls = np.array([5.0])
    vcr = value_capture_ratio(strategy_pnls, oracle_pnls)
    assert vcr == pytest.approx(-2.0)


def test_value_capture_ratio_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        value_capture_ratio(np.array([1.0, 2.0]), np.array([1.0]))
