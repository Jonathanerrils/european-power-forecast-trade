"""Tests for src/uncertainty.py -- the leakage-safety property is the
single most important thing this module has to get right, so it's
tested multiple independent ways, not just via one happy-path example.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.uncertainty import (
    build_continuous_residual_series,
    compute_rolling_residual_quantiles,
    latest_residual_quantile_offsets,
    evaluate_interval_calibration,
    evaluate_interval_calibration_by_fold,
    compute_interval_score,
    evaluate_window_candidate,
    find_common_evaluation_start,
    compute_rolling_coverage_diagnostics,
    summarize_worst_rolling_window,
    compute_residual_quantile_offset,
)


# ---------------------------------------------------------------------
# build_continuous_residual_series
# ---------------------------------------------------------------------
def test_build_continuous_residual_series_computes_residual_correctly():
    fold1 = pd.DataFrame({
        "timestamp_utc": pd.date_range("2023-01-01", periods=3, freq="1h", tz="UTC"),
        "price_eur_mwh": [50.0, 60.0, 70.0],
        "xgboost_full_pred": [48.0, 65.0, 68.0],
    })
    combined = build_continuous_residual_series([fold1], "price_eur_mwh", "xgboost_full_pred")
    assert list(combined["residual"]) == pytest.approx([2.0, -5.0, 2.0])


def test_build_continuous_residual_series_sorts_chronologically_across_folds():
    fold2 = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"),
        "price_eur_mwh": [80.0, 90.0], "pred": [78.0, 88.0],
    })
    fold1 = pd.DataFrame({
        "timestamp_utc": pd.date_range("2023-01-01", periods=2, freq="1h", tz="UTC"),
        "price_eur_mwh": [50.0, 60.0], "pred": [48.0, 58.0],
    })
    # Passed out of order (fold2 before fold1) -- must still sort correctly.
    combined = build_continuous_residual_series([fold2, fold1], "price_eur_mwh", "pred")
    assert combined["timestamp_utc"].is_monotonic_increasing
    assert combined["timestamp_utc"].iloc[0] == pd.Timestamp("2023-01-01", tz="UTC")


def test_build_continuous_residual_series_rejects_overlapping_folds():
    """Regression guard: overlapping fold timestamps would silently
    give one instant two different 'out-of-sample' residuals.
    """
    fold1 = pd.DataFrame({
        "timestamp_utc": pd.date_range("2023-01-01", periods=3, freq="1h", tz="UTC"),
        "price_eur_mwh": [50.0, 60.0, 70.0], "pred": [48.0, 58.0, 68.0],
    })
    fold2 = pd.DataFrame({
        "timestamp_utc": [fold1["timestamp_utc"].iloc[-1]],  # deliberately overlapping
        "price_eur_mwh": [70.0], "pred": [69.0],
    })
    with pytest.raises(ValueError, match="appear in more than one fold"):
        build_continuous_residual_series([fold1, fold2], "price_eur_mwh", "pred")


# ---------------------------------------------------------------------
# compute_rolling_residual_quantiles -- LEAKAGE SAFETY (the core property)
# ---------------------------------------------------------------------
def test_first_row_has_no_prior_residuals_so_forecast_is_nan():
    ts = pd.date_range("2023-01-01", periods=5, freq="1D", tz="UTC")
    residual_series = pd.DataFrame({"timestamp_utc": ts, "prediction": [100.0]*5, "residual": [1.0]*5})
    result = compute_rolling_residual_quantiles(residual_series, [0.5], window_days=3, min_periods_days=1)
    assert pd.isna(result["forecast_q50"].iloc[0])


def test_changing_a_later_residual_never_changes_an_earlier_rows_forecast():
    """THE core leakage-safety test: a row's quantile forecast must be
    a pure function of residuals strictly before it. Mutating a LATER
    residual (even to an extreme, obviously-different value) must
    leave every EARLIER row's forecast_q50 completely unchanged.
    """
    ts = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")
    base = pd.DataFrame({
        "timestamp_utc": ts, "prediction": [100.0]*10,
        "residual": [float(i) for i in range(10)],
    })
    result_before = compute_rolling_residual_quantiles(base, [0.5], window_days=5, min_periods_days=1)

    mutated = base.copy()
    mutated.loc[9, "residual"] = 99999.0  # mutate the LAST row's residual drastically
    result_after = compute_rolling_residual_quantiles(mutated, [0.5], window_days=5, min_periods_days=1)

    # Every row EXCEPT the last (whose own residual was mutated, but that
    # residual is never used for ITS OWN forecast anyway -- closed='left')
    # must be bit-for-bit identical.
    pd.testing.assert_series_equal(
        result_before["forecast_q50"].iloc[:9], result_after["forecast_q50"].iloc[:9]
    )
    # The mutated row's OWN forecast must ALSO be unaffected (closed='left'
    # excludes the current row's own residual from its own window).
    assert result_before["forecast_q50"].iloc[9] == result_after["forecast_q50"].iloc[9]


def test_window_excludes_current_timestamp_exactly() :
    """Manually verified example: residuals 0..9 at days 1..10, window=5
    days. The row at day 6 should see days 1-5 (residuals 0,1,2,3,4,
    median=2.0); the row at day 7 should see days 2-6 (residuals
    1,2,3,4,5, median=3.0) -- NOT including its own day's residual.
    """
    ts = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")
    residual_series = pd.DataFrame({
        "timestamp_utc": ts, "prediction": [100.0]*10,
        "residual": [float(i) for i in range(10)],
    })
    result = compute_rolling_residual_quantiles(residual_series, [0.5], window_days=5, min_periods_days=1)
    assert result["forecast_q50"].iloc[5] == pytest.approx(102.0)  # day 6: median(0..4)=2.0 -> 100+2.0
    assert result["forecast_q50"].iloc[6] == pytest.approx(103.0)  # day 7: median(1..5)=3.0 -> 100+3.0


def test_min_periods_days_produces_nan_during_warmup():
    ts = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")
    residual_series = pd.DataFrame({
        "timestamp_utc": ts, "prediction": [100.0]*10, "residual": [1.0]*10,
    })
    # Require 8 days of history -- rows before day 9 must be NaN.
    result = compute_rolling_residual_quantiles(residual_series, [0.5], window_days=10, min_periods_days=8)
    assert result["forecast_q50"].iloc[:7].isna().all()
    assert result["forecast_q50"].iloc[8:].notna().all()


def test_min_periods_days_out_of_range_raises():
    ts = pd.date_range("2023-01-01", periods=5, freq="1D", tz="UTC")
    residual_series = pd.DataFrame({"timestamp_utc": ts, "prediction": [100.0]*5, "residual": [1.0]*5})
    with pytest.raises(ValueError, match="min_periods_days"):
        compute_rolling_residual_quantiles(residual_series, [0.5], window_days=5, min_periods_days=10)


def test_unsorted_input_is_sorted_before_rolling():
    """Input might not arrive chronologically sorted -- the function
    must sort before computing, not silently roll over a shuffled
    order (which would compute nonsense windows).
    """
    ts = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")
    residual_series = pd.DataFrame({
        "timestamp_utc": ts, "prediction": [100.0]*10, "residual": [float(i) for i in range(10)],
    })
    shuffled = residual_series.sample(frac=1.0, random_state=3).reset_index(drop=True)
    result_sorted = compute_rolling_residual_quantiles(residual_series, [0.5], window_days=5, min_periods_days=1)
    result_shuffled = compute_rolling_residual_quantiles(shuffled, [0.5], window_days=5, min_periods_days=1)
    # Re-sort the shuffled-input result by timestamp before comparing.
    result_shuffled_sorted = result_shuffled.sort_values("timestamp_utc").reset_index(drop=True)
    pd.testing.assert_series_equal(
        result_sorted["forecast_q50"].reset_index(drop=True),
        result_shuffled_sorted["forecast_q50"].reset_index(drop=True),
    )


# ---------------------------------------------------------------------
# latest_residual_quantile_offsets
# ---------------------------------------------------------------------
def test_latest_offsets_match_the_final_rows_own_forecast():
    ts = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")
    residual_series = pd.DataFrame({
        "timestamp_utc": ts, "prediction": [100.0]*10, "residual": [float(i) for i in range(10)],
    })
    offsets = latest_residual_quantile_offsets(residual_series, [0.5], window_days=5, min_periods_days=1)
    full = compute_rolling_residual_quantiles(residual_series, [0.5], window_days=5, min_periods_days=1)
    expected_offset = full["forecast_q50"].iloc[-1] - full["prediction"].iloc[-1]
    assert offsets["q50"] == pytest.approx(expected_offset)


# ---------------------------------------------------------------------
# evaluate_interval_calibration
# ---------------------------------------------------------------------
def test_calibration_reports_perfect_coverage():
    actual = pd.Series([5.0, 5.0, 5.0])
    lower = pd.Series([0.0, 0.0, 0.0])
    upper = pd.Series([10.0, 10.0, 10.0])
    result = evaluate_interval_calibration(actual, lower, upper, nominal_coverage=0.8)
    assert result["empirical_coverage"] == 1.0
    assert result["frac_below_lower"] == 0.0
    assert result["frac_above_upper"] == 0.0


def test_calibration_detects_asymmetric_miscalibration():
    """A single coverage number can hide a badly asymmetric interval --
    this must be caught by the separate below/above fractions.
    """
    actual = pd.Series([100.0, 100.0, 100.0, 5.0])
    lower = pd.Series([0.0, 0.0, 0.0, 0.0])
    upper = pd.Series([10.0, 10.0, 10.0, 10.0])
    result = evaluate_interval_calibration(actual, lower, upper, nominal_coverage=0.8)
    assert result["empirical_coverage"] == 0.25
    assert result["frac_above_upper"] == 0.75
    assert result["frac_below_lower"] == 0.0


def test_calibration_on_known_gaussian_noise_recovers_approximately_correct_coverage():
    """Sanity check with a known ground truth: if residuals are drawn
    from N(0, 1) and we build a 'perfect' 80% interval as the true
    10th/90th percentiles of that same distribution, empirical coverage
    on a large held-out sample should land close to 80%.
    """
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 1, 100_000)
    from scipy.stats import norm
    lo, hi = norm.ppf(0.1), norm.ppf(0.9)
    actual = pd.Series(residuals)
    lower = pd.Series([lo] * len(residuals))
    upper = pd.Series([hi] * len(residuals))
    result = evaluate_interval_calibration(actual, lower, upper, nominal_coverage=0.8)
    assert result["empirical_coverage"] == pytest.approx(0.8, abs=0.01)


def test_calibration_ignores_rows_with_missing_values():
    actual = pd.Series([5.0, np.nan, 5.0])
    lower = pd.Series([0.0, 0.0, 0.0])
    upper = pd.Series([10.0, 10.0, 10.0])
    result = evaluate_interval_calibration(actual, lower, upper, nominal_coverage=0.8)
    assert result["n"] == 2


# ---------------------------------------------------------------------
# Fold tagging + per-fold calibration breakdown
# ---------------------------------------------------------------------
def test_build_continuous_residual_series_tags_fold_when_given():
    fold1 = pd.DataFrame({
        "timestamp_utc": pd.date_range("2023-01-01", periods=2, freq="1h", tz="UTC"),
        "price_eur_mwh": [50.0, 60.0], "pred": [48.0, 58.0],
    })
    fold2 = pd.DataFrame({
        "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"),
        "price_eur_mwh": [80.0, 90.0], "pred": [78.0, 88.0],
    })
    combined = build_continuous_residual_series(
        [fold1, fold2], "price_eur_mwh", "pred", fold_names=["fold_1", "fold_2"]
    )
    assert list(combined["fold"]) == ["fold_1", "fold_1", "fold_2", "fold_2"]


def test_build_continuous_residual_series_fold_names_length_mismatch_raises():
    fold1 = pd.DataFrame({
        "timestamp_utc": pd.date_range("2023-01-01", periods=2, freq="1h", tz="UTC"),
        "price_eur_mwh": [50.0, 60.0], "pred": [48.0, 58.0],
    })
    with pytest.raises(ValueError, match="parallel lists"):
        build_continuous_residual_series([fold1], "price_eur_mwh", "pred", fold_names=["fold_1", "fold_2"])


def test_build_continuous_residual_series_without_fold_names_has_no_fold_column():
    """Backward compatibility: existing callers that don't pass
    fold_names must be completely unaffected.
    """
    fold1 = pd.DataFrame({
        "timestamp_utc": pd.date_range("2023-01-01", periods=2, freq="1h", tz="UTC"),
        "price_eur_mwh": [50.0, 60.0], "pred": [48.0, 58.0],
    })
    combined = build_continuous_residual_series([fold1], "price_eur_mwh", "pred")
    assert "fold" not in combined.columns


def test_per_fold_calibration_isolates_a_badly_miscalibrated_fold():
    """THE core property this function exists for: a badly-miscalibrated
    fold's numbers must not leak into or dilute a well-calibrated
    fold's reported coverage. If they were pooled, a good fold's 100%
    coverage and a bad fold's 0% coverage could average out to look
    like an unremarkable ~50-60%, hiding the real problem.
    """
    df = pd.DataFrame({
        "actual": [5.0, 5.0, 5.0, 100.0, 100.0, 100.0],
        "lower":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "upper":  [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "fold": ["fold_1", "fold_1", "fold_1", "fold_2", "fold_2", "fold_2"],
    })
    result = evaluate_interval_calibration_by_fold(df, "actual", "lower", "upper", nominal_coverage=0.8)
    fold_1_row = result[result["fold"] == "fold_1"].iloc[0]
    fold_2_row = result[result["fold"] == "fold_2"].iloc[0]
    assert fold_1_row["empirical_coverage"] == 1.0   # perfectly calibrated fold stays visible as such
    assert fold_2_row["empirical_coverage"] == 0.0   # badly miscalibrated fold stays visible as such
    assert fold_2_row["frac_above_upper"] == 1.0


def test_per_fold_calibration_preserves_chronological_order_not_alphabetical():
    """regime_stress_test would sort AFTER fold_1/2/3 alphabetically by
    coincidence here, but this must not be relied on -- order should
    reflect first appearance in the data (chronological), not whatever
    groupby's default ordering happens to produce.
    """
    df = pd.DataFrame({
        "actual": [5.0]*6, "lower": [0.0]*6, "upper": [10.0]*6,
        "fold": ["regime_stress_test", "regime_stress_test", "fold_3", "fold_3", "fold_1", "fold_1"],
    })
    result = evaluate_interval_calibration_by_fold(df, "actual", "lower", "upper", nominal_coverage=0.8)
    assert list(result["fold"]) == ["regime_stress_test", "fold_3", "fold_1"]


def test_per_fold_calibration_raises_without_fold_column():
    df = pd.DataFrame({"actual": [5.0], "lower": [0.0], "upper": [10.0]})
    with pytest.raises(ValueError, match="fold_names"):
        evaluate_interval_calibration_by_fold(df, "actual", "lower", "upper", nominal_coverage=0.8)


# ---------------------------------------------------------------------
# compute_interval_score (Winkler/interval score)
# ---------------------------------------------------------------------
def test_interval_score_matches_hand_computed_example():
    """l=0, u=10, alpha=0.2 (80% interval).
    a=5 (inside): score = width = 10.
    a=-5 (below by 5): score = 10 + (2/0.2)*5 = 60.
    a=15 (above by 5): score = 10 + (2/0.2)*5 = 60.
    Mean = (10+60+60)/3 = 43.333...
    """
    actual = pd.Series([5.0, -5.0, 15.0])
    lower = pd.Series([0.0, 0.0, 0.0])
    upper = pd.Series([10.0, 10.0, 10.0])
    score = compute_interval_score(actual, lower, upper, alpha=0.2)
    assert score == pytest.approx((10 + 60 + 60) / 3)


def test_interval_score_is_pure_width_when_always_inside():
    actual = pd.Series([1.0, 5.0, 9.0])
    lower = pd.Series([0.0, 0.0, 0.0])
    upper = pd.Series([10.0, 10.0, 10.0])
    score = compute_interval_score(actual, lower, upper, alpha=0.2)
    assert score == pytest.approx(10.0)


def test_interval_score_cannot_be_gamed_by_widening_alone():
    """A wider interval that eliminates misses is not automatically
    better -- the width penalty accrues on every row, not just misses.
    This proves the score resists the exact failure mode
    evaluate_interval_calibration (coverage alone) is vulnerable to.
    """
    actual = pd.Series([5.0] * 100)  # always comfortably inside both intervals below
    narrow_score = compute_interval_score(actual, pd.Series([0.0]*100), pd.Series([10.0]*100), alpha=0.2)
    wide_score = compute_interval_score(actual, pd.Series([-1000.0]*100), pd.Series([1000.0]*100), alpha=0.2)
    assert narrow_score < wide_score  # narrower interval wins when both fully cover


def test_interval_score_cannot_be_gamed_by_narrowing_alone():
    """A narrower interval that causes more misses is not automatically
    better either -- large miss penalties (scaled by 2/alpha) dominate
    the small width saving.
    """
    actual = pd.Series([-50.0, 5.0, 50.0])  # extreme values likely to miss a narrow interval
    narrow_score = compute_interval_score(actual, pd.Series([4.0]*3), pd.Series([6.0]*3), alpha=0.2)
    wide_score = compute_interval_score(actual, pd.Series([-100.0]*3), pd.Series([100.0]*3), alpha=0.2)
    assert wide_score < narrow_score  # here, wide (covers extremes) beats narrow (misses badly)


# ---------------------------------------------------------------------
# evaluate_window_candidate -- the sensitivity experiment's core unit
# ---------------------------------------------------------------------
def test_evaluate_window_candidate_returns_all_required_fields():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2023-01-01", periods=24 * 300, freq="1h", tz="UTC")
    residual_series = pd.DataFrame({
        "timestamp_utc": ts,
        "price_eur_mwh": 80 + rng.normal(0, 10, len(ts)),
        "prediction": [80.0] * len(ts),
        "fold": ["fold_1"] * len(ts),
    })
    residual_series["residual"] = residual_series["price_eur_mwh"] - residual_series["prediction"]

    result = evaluate_window_candidate(
        residual_series, [0.1, 0.5, 0.9], window_days=90, min_periods_days=22, target_col="price_eur_mwh"
    )
    assert result["window_days"] == 90
    assert result["pooled_calibration"]["n"] > 0
    assert result["by_fold_calibration"] is not None
    assert result["mean_width"] > 0
    assert result["interval_score"] > 0
    assert result["n_warmup"] > 0  # some warm-up rows expected given 90-day window


def test_evaluate_window_candidate_narrower_window_has_fewer_warmup_rows_than_wider():
    """A shorter window needs less history before producing its first
    non-NaN interval -- a basic sanity check that window_days is
    actually driving the warm-up length, not being ignored.
    """
    rng = np.random.default_rng(1)
    ts = pd.date_range("2023-01-01", periods=24 * 300, freq="1h", tz="UTC")
    residual_series = pd.DataFrame({
        "timestamp_utc": ts,
        "price_eur_mwh": 80 + rng.normal(0, 10, len(ts)),
        "prediction": [80.0] * len(ts),
    })
    residual_series["residual"] = residual_series["price_eur_mwh"] - residual_series["prediction"]

    short = evaluate_window_candidate(residual_series, [0.1, 0.9], window_days=60, min_periods_days=15, target_col="price_eur_mwh")
    long = evaluate_window_candidate(residual_series, [0.1, 0.9], window_days=180, min_periods_days=45, target_col="price_eur_mwh")
    assert short["n_warmup"] < long["n_warmup"]


# ---------------------------------------------------------------------
# Common-row-set evaluation for the sensitivity comparison
#
# Regression context: candidates with different window_days have
# different warm-up lengths by construction, so without a common-row
# correction, a shorter window is partly scored on early data a longer
# window never gets evaluated on at all -- confounding "which window
# is better" with "which period got included." Same principle already
# enforced for point-model comparison elsewhere in this project
# (baseline_v1's common_comparison_rows guarantee).
# ---------------------------------------------------------------------
def _synthetic_residual_series(periods=24 * 400, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=periods, freq="1h", tz="UTC")
    price = 80 + rng.normal(0, 10, periods)
    residual_series = pd.DataFrame({
        "timestamp_utc": ts, "price_eur_mwh": price, "prediction": [80.0] * periods,
    })
    residual_series["residual"] = residual_series["price_eur_mwh"] - residual_series["prediction"]
    return residual_series


def test_find_common_evaluation_start_is_the_latest_of_all_warmups():
    residual_series = _synthetic_residual_series()
    configs = [(60, 15), (180, 45), (365, 91)]
    common_start = find_common_evaluation_start(residual_series, [0.1, 0.5, 0.9], configs)

    # Cross-check: the 365-day candidate (longest warm-up) alone should
    # produce this exact same cutoff, since it's the binding constraint.
    qf_365 = compute_rolling_residual_quantiles(residual_series, [0.1, 0.5, 0.9], 365, 91)
    expected = qf_365.loc[qf_365["forecast_q10"].notna(), "timestamp_utc"].min()
    assert common_start == expected


def test_common_row_evaluation_gives_every_candidate_identical_n():
    """THE regression test the common-row correction exists for: after
    applying common_start, every candidate in a comparison must be
    scored on the exact same number of rows -- not just similar,
    identical.
    """
    residual_series = _synthetic_residual_series()
    quantiles = [0.1, 0.5, 0.9]
    configs = [(60, 15), (90, 22), (120, 30), (180, 45), (365, 91)]

    common_start = find_common_evaluation_start(residual_series, quantiles, configs)
    ns = []
    for window_days, min_periods_days in configs:
        result = evaluate_window_candidate(
            residual_series, quantiles, window_days, min_periods_days,
            target_col="price_eur_mwh", common_start=common_start,
        )
        ns.append(result["pooled_calibration"]["n"])
    assert len(set(ns)) == 1, f"candidates were NOT scored on identical row counts: {ns}"


def test_without_common_start_candidates_have_different_n():
    """Confirms the correction actually does something -- without it,
    different window_days genuinely DO produce different n (this is
    the exact confound the correction removes).
    """
    residual_series = _synthetic_residual_series()
    quantiles = [0.1, 0.5, 0.9]
    n_60 = evaluate_window_candidate(residual_series, quantiles, 60, 15, target_col="price_eur_mwh")["pooled_calibration"]["n"]
    n_365 = evaluate_window_candidate(residual_series, quantiles, 365, 91, target_col="price_eur_mwh")["pooled_calibration"]["n"]
    assert n_60 != n_365


def test_evaluate_window_candidate_without_common_start_is_unchanged():
    """Backward compatibility: omitting common_start must behave
    exactly as before this feature was added.
    """
    residual_series = _synthetic_residual_series()
    result = evaluate_window_candidate(
        residual_series, [0.1, 0.5, 0.9], window_days=90, min_periods_days=22, target_col="price_eur_mwh"
    )
    assert result["n_warmup"] > 0
    assert result["pooled_calibration"]["n"] > 0


def test_evaluate_window_candidate_return_per_row_false_by_default():
    """Existing callers (the sensitivity experiment) must not pay for
    or receive per-row data unless they explicitly opt in.
    """
    residual_series = _synthetic_residual_series()
    result = evaluate_window_candidate(
        residual_series, [0.1, 0.5, 0.9], window_days=90, min_periods_days=22, target_col="price_eur_mwh"
    )
    assert "per_row" not in result


def test_evaluate_window_candidate_return_per_row_true_gives_usable_bounds():
    """Needed for run_uncertainty_tier1_robustness.py to actually SAVE
    Tier-1's per-timestamp quantile bounds -- previously computed
    internally and silently discarded, leaving no artifact for a later
    strategy backtest to read per the frozen provenance rule.
    """
    residual_series = _synthetic_residual_series()
    result = evaluate_window_candidate(
        residual_series, [0.1, 0.5, 0.9], window_days=90, min_periods_days=22,
        target_col="price_eur_mwh", return_per_row=True,
    )
    assert "per_row" in result
    per_row = result["per_row"]
    assert "timestamp_utc" in per_row.columns
    assert "forecast_q10" in per_row.columns
    assert "forecast_q90" in per_row.columns
    assert len(per_row) == len(residual_series)
    # The saved bounds must be actual PRICE-scale numbers (point forecast
    # +/- offset), consistent with the pooled calibration computed from
    # the exact same columns -- not some other, disconnected quantity.
    valid = per_row["forecast_q10"].notna()
    assert (per_row.loc[valid, "forecast_q10"] < per_row.loc[valid, "forecast_q90"]).all()


# ---------------------------------------------------------------------
# Rolling coverage diagnostics -- DIAGNOSTIC ONLY, never a selection metric
# ---------------------------------------------------------------------
def test_rolling_coverage_surfaces_a_bad_stretch_a_pooled_number_hides():
    """THE core property this diagnostic exists for: a pooled coverage
    number can look moderately concerning (66.7%) while completely
    hiding that one specific 30-day stretch was a total failure (0%
    coverage). The worst rolling window must surface that, not average
    it away.
    """
    rng = np.random.default_rng(0)
    ts = pd.date_range("2023-01-01", periods=24 * 90, freq="1h", tz="UTC")
    n = len(ts)
    actual = rng.uniform(-5, 5, n)
    lower = np.full(n, -5.0)
    upper = np.full(n, 5.0)
    bad_start, bad_end = 24 * 30, 24 * 60
    actual[bad_start:bad_end] = rng.uniform(50, 100, bad_end - bad_start)

    diag = compute_rolling_coverage_diagnostics(pd.Series(actual), pd.Series(lower), pd.Series(upper), pd.Series(ts), window_days=30)
    worst = summarize_worst_rolling_window(diag, nominal_coverage=1.0)
    assert worst["worst_coverage"] == pytest.approx(0.0, abs=1e-9)


def test_rolling_coverage_is_point_in_time_closed_right_by_design():
    """This is a RETROSPECTIVE diagnostic (never fed back into any
    interval computation), so closed='right' (including the current
    row) is correct here -- unlike compute_rolling_residual_quantiles,
    which must exclude the current row because its output IS used to
    build that row's own forecast. Confirms the window at the LAST
    timestamp includes that timestamp's own outcome.
    """
    ts = pd.date_range("2023-01-01", periods=5, freq="1D", tz="UTC")
    actual = pd.Series([0.0, 0.0, 0.0, 0.0, 100.0])  # last value is a miss
    lower = pd.Series([-1.0] * 5)
    upper = pd.Series([1.0] * 5)
    diag = compute_rolling_coverage_diagnostics(actual, lower, upper, pd.Series(ts), window_days=10)
    # The last row's own miss must be reflected in its own rolling coverage.
    assert diag["rolling_coverage"].iloc[-1] < 1.0


def test_rolling_coverage_flat_perfect_series_reports_full_coverage():
    ts = pd.date_range("2023-01-01", periods=100, freq="1D", tz="UTC")
    actual = pd.Series([0.0] * 100)
    lower = pd.Series([-1.0] * 100)
    upper = pd.Series([1.0] * 100)
    diag = compute_rolling_coverage_diagnostics(actual, lower, upper, pd.Series(ts), window_days=30)
    assert (diag["rolling_coverage"].dropna() == 1.0).all()


def test_worst_rolling_window_reflects_genuine_contamination_not_just_warmup_noise():
    """A single early miss legitimately degrades every 30-day trailing
    window that still contains it (for up to window_days after it
    occurs) -- that's correct rolling-window behavior, not warm-up
    noise to filter out. This checks the more precise property: once
    the contaminated point has fully rolled OUT of the window (more
    than window_days after it occurred), coverage must return to
    perfect, proving the single-point miss doesn't permanently
    contaminate the whole series.
    """
    ts = pd.date_range("2023-01-01", periods=200, freq="1D", tz="UTC")
    actual = pd.Series([100.0] + [0.0] * 199)  # first observation alone is a dramatic miss
    lower = pd.Series([-1.0] * 200)
    upper = pd.Series([1.0] * 200)
    diag = compute_rolling_coverage_diagnostics(actual, lower, upper, pd.Series(ts), window_days=30)
    # Well past the point where day 0's miss has rolled out of every
    # window (window_days=30, so day 60+ is unambiguously clear of it).
    late_coverage = diag[diag["timestamp_utc"] >= pd.Timestamp("2023-03-15", tz="UTC")]["rolling_coverage"]
    assert (late_coverage == 1.0).all()


# ---------------------------------------------------------------------
# compute_residual_quantile_offset -- needed to recombine ONE model's
# point forecast with a DIFFERENT model's uncertainty envelope width
# ---------------------------------------------------------------------
def test_offset_plus_prediction_reconstructs_the_original_forecast_exactly():
    """offset = forecast_qXX - prediction, so adding it back to the
    SAME series's own prediction must reconstruct forecast_qXX exactly
    -- this is the property the whole point/envelope recombination
    trick depends on being trustworthy.
    """
    ts = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")
    residual_series = pd.DataFrame({
        "timestamp_utc": ts, "prediction": [100.0] * 10, "residual": [float(i) for i in range(10)],
    })
    offset = compute_residual_quantile_offset(residual_series, 0.5, window_days=5, min_periods_days=1)
    full = compute_rolling_residual_quantiles(residual_series, [0.5], window_days=5, min_periods_days=1)
    reconstructed = full["prediction"] + offset["offset"]
    pd.testing.assert_series_equal(reconstructed.rename("forecast_q50"), full["forecast_q50"], check_names=True)


def test_offset_can_be_added_to_a_different_series_point_forecast():
    """The actual intended use: apply series A's offset to series B's
    prediction. Hand-verified: A's offset at a fully-warmed-up row is
    the rolling median of A's own past residuals; adding it to B's
    (different) prediction should NOT reduce to A's or B's own
    self-consistent forecast -- it's a genuinely new, hybrid quantity.
    """
    ts = pd.date_range("2023-01-01", periods=10, freq="1D", tz="UTC")
    series_a = pd.DataFrame({
        "timestamp_utc": ts, "prediction": [100.0] * 10, "residual": [1.0] * 10,  # constant residual = 1.0
    })
    series_b_prediction = pd.Series([200.0] * 10)

    offset_a = compute_residual_quantile_offset(series_a, 0.5, window_days=5, min_periods_days=1)
    hybrid = series_b_prediction + offset_a["offset"]
    # Once warmed up, A's residual is always 1.0, so A's rolling median offset is exactly 1.0.
    assert hybrid.iloc[-1] == pytest.approx(200.0 + 1.0)
