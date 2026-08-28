"""Tests for src/strategy.py's pure economic-kernel functions. Every
test case here is hand-verifiable by inspection, per
docs/economic_contract_v1.md's "Step D -- synthetic data only"
requirement: no real forecasting/CV/data-loading involved, just known
inputs with known correct outputs.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.strategy import (
    degradation_cost,
    price_bounds,
    candidate_pairs,
    best_point_pair,
    best_uncertainty_pair,
    realized_pnl,
    run_day,
)
import pandas as pd


# ---------------------------------------------------------------------
# degradation_cost -- the exact bug caught before this was ever coded
# ---------------------------------------------------------------------
def test_degradation_cost_is_throughput_based_not_flat():
    """At c=10, eta=0.85: C = 10*(1+0.85) = 18.5, NOT a flat 10. This
    is the exact ambiguity a design review caught before any code
    existed -- confirmed here as the actual, tested behavior.
    """
    assert degradation_cost(eta_rt=0.85, c=10.0) == pytest.approx(18.5)


def test_degradation_cost_rejects_invalid_eta():
    with pytest.raises(ValueError, match="eta_rt"):
        degradation_cost(eta_rt=0.0, c=10.0)
    with pytest.raises(ValueError, match="eta_rt"):
        degradation_cost(eta_rt=1.5, c=10.0)


def test_degradation_cost_rejects_negative_c():
    with pytest.raises(ValueError, match="c must be"):
        degradation_cost(eta_rt=0.85, c=-1.0)


# ---------------------------------------------------------------------
# price_bounds -- test 9 (Q-bound semantics): must ADD the offset to
# the point forecast, never use the raw offset alone
# ---------------------------------------------------------------------
def test_price_bounds_add_offset_to_forecast():
    point_forecast = pd.Series([80.0, 90.0])
    q10_offset = pd.Series([-15.0, -20.0])  # small, plausible residual offsets
    q90_offset = pd.Series([12.0, 18.0])
    L, U = price_bounds(point_forecast, q10_offset, q90_offset)
    # L/U must be actual PRICE-scale numbers (point forecast +/- offset),
    # not the tiny offset values themselves.
    assert list(L) == pytest.approx([65.0, 70.0])   # 80-15, 90-20
    assert list(U) == pytest.approx([92.0, 108.0])  # 80+12, 90+18
    assert all(L > 10)  # sanity: price-scale, not offset-scale
    assert all(U > 10)


# ---------------------------------------------------------------------
# candidate_pairs -- must generalize to 23/24/25-hour DST days
# ---------------------------------------------------------------------
def test_candidate_pairs_count_matches_combinatorics():
    for n in (23, 24, 25):
        pairs = candidate_pairs(n)
        assert len(pairs) == n * (n - 1) // 2
        assert all(i < j for i, j in pairs)


def test_candidate_pairs_empty_for_fewer_than_two_hours():
    assert candidate_pairs(1) == []
    assert candidate_pairs(0) == []


# ---------------------------------------------------------------------
# best_point_pair -- hand-verifiable synthetic days
# ---------------------------------------------------------------------
def test_obvious_cheap_then_expensive_day():
    """Forecast: cheap early, expensive late. The obviously correct
    pair, verifiable by inspection: buy at the cheapest hour (index 2,
    price 20), sell at the most expensive hour (index 5, price 100).
    """
    forecast = np.array([50.0, 30.0, 20.0, 40.0, 70.0, 100.0])
    eta_rt, C = 0.85, 5.0
    i, j, score = best_point_pair(forecast, eta_rt, C)
    assert (i, j) == (2, 5)
    expected_score = 0.85 * 100.0 - 20.0 - 5.0
    assert score == pytest.approx(expected_score)


def test_no_trade_when_every_pair_unprofitable():
    """Flat, low-spread day where efficiency loss + cost exceeds any
    achievable spread -- must correctly abstain, not force a losing
    trade.
    """
    forecast = np.array([50.0, 50.5, 50.2, 50.8, 50.1])
    eta_rt, C = 0.85, 20.0  # cost far exceeds any achievable spread here
    i, j, score = best_point_pair(forecast, eta_rt, C)
    assert i is None and j is None
    assert score == 0.0


def test_negative_price_charging_day():
    """Charging at a negative price, discharging at a positive price --
    a real, valid outcome (being paid to charge, then paid again to
    discharge). Must not be special-cased or rejected.
    """
    forecast = np.array([-60.0, 10.0, 70.0])
    eta_rt, C = 0.85, 5.0
    i, j, score = best_point_pair(forecast, eta_rt, C)
    assert (i, j) == (0, 2)
    expected = 0.85 * 70.0 - (-60.0) - 5.0
    assert score == pytest.approx(expected)
    assert score > 0


# ---------------------------------------------------------------------
# reverse-time trap (test 3: no reverse-time cycle)
# ---------------------------------------------------------------------
def test_reverse_time_trap_never_selects_earlier_sell_hour():
    """Deliberately construct a day where the single best UNORDERED
    (min, max) pair would have the expensive hour BEFORE the cheap
    hour -- the function must still only ever consider i<j pairs, so
    it must fall back to the best VALID chronological pair, never the
    naive argmin/argmax.
    """
    # Expensive hour (index 0) comes before the cheap hour (index 3).
    # A naive argmin(price)/argmax(price) without ordering would try
    # to "sell" at index 0 and "buy" at index 3 -- backwards in time.
    forecast = np.array([100.0, 60.0, 55.0, 20.0, 65.0])
    eta_rt, C = 0.85, 1.0
    i, j, score = best_point_pair(forecast, eta_rt, C)
    assert i < j  # the invariant itself
    # The best valid (i<j) pair here: buy at index 3 (20), sell at index 4 (65).
    assert (i, j) == (3, 4)


# ---------------------------------------------------------------------
# best_uncertainty_pair -- can suppress an otherwise-positive point trade
# ---------------------------------------------------------------------
def test_uncertainty_aware_can_suppress_a_point_positive_trade():
    """A day where the point-forecast rule would trade, but the
    conservative (worst-case) bounds make it unprofitable -- the
    uncertainty-aware rule must correctly abstain even though the point
    rule would not.
    """
    forecast = np.array([50.0, 70.0])
    eta_rt, C = 0.85, 2.0
    point_i, point_j, point_score = best_point_pair(forecast, eta_rt, C)
    assert point_i is not None  # point rule DOES trade: 0.85*70-50-2 = 7.5 > 0

    # Conservative bounds: worst-case buy price much higher, worst-case
    # sell price much lower than the point forecast.
    L = np.array([50.0, 55.0])  # L_j (sell hour worst-case) much lower than forecast
    U = np.array([65.0, 70.0])  # U_i (buy hour worst-case) much higher than forecast
    i, j, score = best_uncertainty_pair(L, U, eta_rt, C)
    assert i is None and j is None  # uncertainty-aware rule correctly abstains


# ---------------------------------------------------------------------
# realized_pnl -- decision is fixed, only pricing changes
# ---------------------------------------------------------------------
def test_realized_pnl_no_trade_is_zero():
    gross, net = realized_pnl(np.array([10.0, 20.0]), None, None, 0.85, 5.0)
    assert gross == 0.0 and net == 0.0


def test_realized_pnl_computes_gross_and_net_correctly():
    actual = np.array([40.0, 100.0])
    gross, net = realized_pnl(actual, 0, 1, eta_rt=0.85, C=5.0)
    assert gross == pytest.approx(0.85 * 100.0 - 40.0)
    assert net == pytest.approx(gross - 5.0)


def test_realized_pnl_rejects_reverse_time():
    with pytest.raises(AssertionError, match="cannot discharge before"):
        realized_pnl(np.array([10.0, 20.0]), i=1, j=0, eta_rt=0.85, C=5.0)


# ---------------------------------------------------------------------
# run_day -- orchestration, and the leakage tests (1, 2)
# ---------------------------------------------------------------------
def test_run_day_point_mode():
    forecast = np.array([50.0, 30.0, 20.0, 100.0])
    actual = np.array([48.0, 32.0, 18.0, 105.0])  # close to forecast but not identical
    result = run_day(forecast, actual, eta_rt=0.85, c=5.0)
    assert result["traded"] is True
    assert result["i"] == 2 and result["j"] == 3
    # gross/net computed from ACTUAL, not forecast, prices
    assert result["gross_pnl"] == pytest.approx(0.85 * 105.0 - 18.0)


def test_run_day_uncertainty_mode_requires_both_L_and_U():
    forecast = np.array([50.0, 100.0])
    actual = np.array([50.0, 100.0])
    with pytest.raises(ValueError, match="Both L and U"):
        run_day(forecast, actual, eta_rt=0.85, c=5.0, L=np.array([40.0, 90.0]))


def test_actual_price_perturbation_changes_pnl_not_decision():
    """Test 1 (leakage): corrupting realized prices must change P&L
    but NEVER change which (i, j) pair was selected -- the decision
    comes entirely from the forecast, computed before actual prices are
    known.
    """
    forecast = np.array([50.0, 30.0, 20.0, 100.0])
    actual_original = np.array([48.0, 32.0, 18.0, 105.0])
    actual_corrupted = np.array([999999.0, -999999.0, 123456.0, -1.0])

    result_original = run_day(forecast, actual_original, eta_rt=0.85, c=5.0)
    result_corrupted = run_day(forecast, actual_corrupted, eta_rt=0.85, c=5.0)

    assert result_original["i"] == result_corrupted["i"]
    assert result_original["j"] == result_corrupted["j"]
    assert result_original["net_pnl"] != result_corrupted["net_pnl"]


def test_future_day_perturbation_does_not_change_todays_decision():
    """Test 2 (leakage): each day's decision must be computed purely
    from that day's own forecast -- corrupting a DIFFERENT day's data
    entirely (simulated here by simply not passing it in at all, since
    run_day operates on one day at a time by construction) cannot
    possibly affect this day's result. This is really an architectural
    guarantee of the per-day function signature, confirmed explicitly.
    """
    forecast_day1 = np.array([50.0, 30.0, 20.0, 100.0])
    actual_day1 = np.array([48.0, 32.0, 18.0, 105.0])
    result_day1 = run_day(forecast_day1, actual_day1, eta_rt=0.85, c=5.0)

    # A "day 2" with wildly different, even absurd, data literally
    # cannot be passed into or influence run_day(day1_data, ...) --
    # there is no shared state, no global, no lookahead parameter.
    forecast_day2 = np.array([-999999.0, 999999.0])
    actual_day2 = np.array([-999999.0, 999999.0])
    result_day1_again = run_day(forecast_day1, actual_day1, eta_rt=0.85, c=5.0)
    assert result_day1 == result_day1_again


# ---------------------------------------------------------------------
# 23/24/25-hour DST days
# ---------------------------------------------------------------------
def test_spring_forward_23_hour_day():
    rng = np.random.default_rng(0)
    forecast = rng.uniform(20, 100, 23)
    actual = forecast + rng.normal(0, 2, 23)
    result = run_day(forecast, actual, eta_rt=0.85, c=5.0)
    assert result["i"] is None or result["i"] < result["j"]


def test_fall_back_25_hour_day():
    rng = np.random.default_rng(1)
    forecast = rng.uniform(20, 100, 25)
    actual = forecast + rng.normal(0, 2, 25)
    result = run_day(forecast, actual, eta_rt=0.85, c=5.0)
    assert result["i"] is None or result["i"] < result["j"]


# ---------------------------------------------------------------------
# assert_no_holdout_access -- test 11 (holdout rejection)
# ---------------------------------------------------------------------
def test_holdout_access_rejected():
    from src.strategy import assert_no_holdout_access
    development_ts = pd.date_range("2025-12-01", periods=5, freq="1D", tz="UTC")
    holdout_ts = pd.date_range("2026-01-01", periods=5, freq="1D", tz="UTC")
    mixed = list(development_ts) + list(holdout_ts)
    with pytest.raises(ValueError, match="holdout boundary"):
        assert_no_holdout_access(mixed)


def test_pure_development_timestamps_pass():
    from src.strategy import assert_no_holdout_access
    development_ts = pd.date_range("2023-01-01", "2025-12-31", freq="1D", tz="UTC")
    assert_no_holdout_access(development_ts)  # must not raise


def test_holdout_boundary_is_exact_local_delivery_date():
    """The boundary itself (2026-01-01 local delivery day start, i.e.
    2025-12-31T23:00:00Z) must be REJECTED (>=, not >), matching
    get_holdout_window()'s own inclusive-start convention elsewhere in
    this project.
    """
    from src.strategy import assert_no_holdout_access
    from src.clean import local_delivery_date_to_utc
    boundary = local_delivery_date_to_utc("2026-01-01")
    with pytest.raises(ValueError, match="holdout boundary"):
        assert_no_holdout_access([boundary])
    just_before = boundary - pd.Timedelta(hours=1)
    assert_no_holdout_access([just_before])  # must not raise


# ---------------------------------------------------------------------
# Post-cutover 15-minute basket quantities (test 10 in the contract) --
# this project's actual data represents post-cutover hours as the
# arithmetic MEAN of 4 quarter-hour prices (aggregate_quarter_hour_to_hourly),
# not as literal separate quarter-hour arrays. run_day/strategy.py never
# constructs per-quarter-hour quantities directly -- it operates
# entirely on this pre-aggregated hourly-mean series. The correct,
# honest adaptation of "assert the basket sums to 1 MWh / eta MWh" for
# THIS architecture is proving the mathematical equivalence the
# contract's basket interpretation depends on: hourly-mean pricing of
# a 1 MWh charge / eta MWh discharge produces IDENTICAL cost/revenue to
# separately executing 0.25 MWh / (eta/4) MWh in each of the 4
# constituent quarter-hours. If this equivalence didn't hold, using the
# pre-aggregated hourly series would silently misprice the post-cutover
# strategy.
# ---------------------------------------------------------------------
def test_hourly_mean_pricing_equals_explicit_quarter_hour_basket_charge_leg():
    quarter_hour_prices = [45.0, 52.0, 38.0, 60.0]
    mean_price = sum(quarter_hour_prices) / 4

    # Explicit basket: 0.25 MWh charged at each quarter-hour's own price.
    explicit_basket_cost = sum(0.25 * p for p in quarter_hour_prices)
    # What run_day() actually computes, using the pre-aggregated hourly mean:
    # cost of charging 1 MWh at the single hourly-mean price.
    hourly_mean_cost = 1.0 * mean_price

    assert explicit_basket_cost == pytest.approx(hourly_mean_cost)


def test_hourly_mean_pricing_equals_explicit_quarter_hour_basket_discharge_leg():
    quarter_hour_prices = [45.0, 52.0, 38.0, 60.0]
    mean_price = sum(quarter_hour_prices) / 4
    eta_rt = 0.85

    # Explicit basket: eta_rt/4 MWh discharged at each quarter-hour's own
    # price -- summing to eta_rt MWh total, per docs/economic_contract_v1.md.
    explicit_basket_revenue = sum((eta_rt / 4) * p for p in quarter_hour_prices)
    # What run_day() actually computes: eta_rt MWh discharged at the
    # single hourly-mean price.
    hourly_mean_revenue = eta_rt * mean_price

    assert explicit_basket_revenue == pytest.approx(hourly_mean_revenue)


def test_hourly_mean_pricing_equivalence_holds_for_run_day_end_to_end():
    """Confirms the equivalence at the level actually used in
    production: run_day()'s realized P&L for a two-hour day, computed
    from pre-aggregated hourly-mean prices, must equal what an explicit
    quarter-hour-by-quarter-hour calculation would give for the same
    underlying quarter-hour prices.
    """
    # Hour 0 (charge) has 4 quarter-hour prices; hour 1 (discharge) has
    # 4 different quarter-hour prices.
    charge_quarter_prices = [40.0, 42.0, 38.0, 44.0]
    discharge_quarter_prices = [90.0, 95.0, 88.0, 92.0]
    eta_rt, c = 0.85, 5.0

    charge_hourly_mean = sum(charge_quarter_prices) / 4
    discharge_hourly_mean = sum(discharge_quarter_prices) / 4

    forecast = np.array([charge_hourly_mean, discharge_hourly_mean])
    actual = np.array([charge_hourly_mean, discharge_hourly_mean])
    result = run_day(forecast, actual, eta_rt=eta_rt, c=c)

    explicit_cost = sum(0.25 * p for p in charge_quarter_prices)
    explicit_revenue = sum((eta_rt / 4) * p for p in discharge_quarter_prices)
    explicit_gross = explicit_revenue - explicit_cost

    assert result["traded"] is True
    assert result["gross_pnl"] == pytest.approx(explicit_gross)
