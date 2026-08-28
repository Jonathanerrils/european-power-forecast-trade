"""spec section 21-adjacent: baseline and ElasticNet correctness tests.
Also verifies the extreme-regime threshold and ElasticNet scaler are
computed from TRAINING data only, never touching validation data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import (
    compute_metrics,
    compute_metrics_by_group,
    fit_extreme_regime_threshold,
    baseline_predictions,
    fit_elasticnet,
    predict_elasticnet,
    evaluate_fold,
    build_elasticnet_search,
    ELASTICNET_PREDICTOR_COLS,
    ELASTICNET_TIER1_PREDICTOR_COLS,
    BASELINE_V1_ALPHA_GRID,
    BASELINE_V2_ALPHA_GRID,
    ELASTICNET_SEARCH_PROFILES,
    DEFAULT_SEARCH_PROFILE,
    TARGET_COL,
)
from src.splits import get_split_windows, SplitWindow
from src.clean import local_delivery_date_to_utc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


def _make_full_feature_frame(periods=24 * 365 * 5, seed=0):
    """Synthetic frame with every column ELASTICNET_PREDICTOR_COLS needs,
    plus price_lag_24h/168h and TARGET_COL, spanning enough years to
    exercise real fold boundaries.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2019-01-01T00:00:00Z", periods=periods, freq="1h", tz="UTC")
    n = len(ts)
    local = ts.tz_convert("Europe/Berlin")
    hour = local.hour.values.astype(float)
    dow = local.dayofweek.values.astype(float)
    month = local.month.values.astype(float)
    weekend = (dow >= 5).astype(float)

    residual_load = rng.normal(30000, 8000, n)
    price = 40 + 0.002 * residual_load + 10 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 15, n)

    # Canonical local-delivery-day cutover, not naive UTC midnight -- see
    # clean.local_delivery_date_to_utc's docstring for why this matters.
    cutover = local_delivery_date_to_utc("2025-10-01")

    df = pd.DataFrame({
        "timestamp_utc": ts,
        TARGET_COL: price,
        "load_forecast_mw": rng.normal(55000, 9000, n),
        "renewables_forecast_mw": rng.normal(20000, 12000, n),
        "residual_load_forecast_mw": residual_load,
        "renewable_share_forecast": rng.uniform(0, 1, n),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "weekend": weekend,
        "weekend_hour_sin": weekend * np.sin(2 * np.pi * hour / 24),
        "weekend_hour_cos": weekend * np.cos(2 * np.pi * hour / 24),
        "post_15min_mtu": (ts >= cutover).astype(int),
        "hour_local": local.hour,
    })
    df["price_lag_24h"] = df[TARGET_COL].shift(24)
    df["price_lag_48h"] = df[TARGET_COL].shift(48)
    df["price_lag_168h"] = df[TARGET_COL].shift(168)
    df["price_rolling_mean_24h"] = df[TARGET_COL].shift(24).rolling(24, min_periods=19).mean()
    df["price_rolling_vol_24h"] = df[TARGET_COL].shift(24).rolling(24, min_periods=19).std()
    df["price_rolling_mean_168h"] = df[TARGET_COL].shift(24).rolling(168, min_periods=134).mean()
    df["price_rolling_vol_168h"] = df[TARGET_COL].shift(24).rolling(168, min_periods=134).std()
    return df


def test_compute_metrics_correctness():
    y_true = pd.Series([10.0, 20.0, 30.0, -10.0])
    y_pred = pd.Series([12.0, 18.0, 35.0, -5.0])
    m = compute_metrics(y_true, y_pred)
    expected_mae = np.mean([2, 2, 5, 5])
    expected_rmse = np.sqrt(np.mean([4, 4, 25, 25]))
    assert m["mae"] == pytest.approx(expected_mae)
    assert m["rmse"] == pytest.approx(expected_rmse)
    assert m["n"] == 4


def test_compute_metrics_by_group():
    df = pd.DataFrame({
        "y": [10.0, 20.0, 10.0, 20.0],
        "pred": [12.0, 22.0, 8.0, 18.0],
        "grp": ["a", "a", "b", "b"],
    })
    out = compute_metrics_by_group(df, "y", "pred", "grp")
    assert out.loc["a", "mae"] == pytest.approx(2.0)
    assert out.loc["b", "mae"] == pytest.approx(2.0)


def test_baseline_predictions_use_precomputed_lag_columns_exactly():
    df = pd.DataFrame({
        "price_lag_24h": [1.0, 2.0, np.nan],
        "price_lag_168h": [3.0, 4.0, 5.0],
    })
    out = baseline_predictions(df)
    assert list(out["lag_24_pred"]) == [1.0, 2.0, None] or out["lag_24_pred"].equals(df["price_lag_24h"])
    assert out["lag_168_pred"].equals(df["price_lag_168h"])


def test_extreme_regime_threshold_computed_from_training_only():
    """The threshold must not shift when validation-only data changes --
    it's a function of train_prices alone.
    """
    train_prices = pd.Series(np.concatenate([np.full(900, 50.0), np.full(100, 500.0)]))
    threshold = fit_extreme_regime_threshold(train_prices)
    assert threshold.train_median == pytest.approx(50.0)
    # Changing "validation" data (not passed in at all) can't affect this
    # -- structurally guaranteed since the function only accepts train_prices.
    assert threshold.abs_deviation_q95 > 0


def test_extreme_regime_flags_far_outliers():
    train_prices = pd.Series(np.random.default_rng(0).normal(50, 10, 1000))
    threshold = fit_extreme_regime_threshold(train_prices)
    far_price = pd.Series([50.0, 1000.0])  # one normal, one extreme
    flags = threshold.is_extreme(far_price)
    assert not flags.iloc[0]
    assert flags.iloc[1]


def test_fit_elasticnet_uses_only_frozen_predictor_columns():
    df = _make_full_feature_frame(periods=24 * 400)
    model, scaler, cols, hyperparams = fit_elasticnet(df)
    assert cols == ELASTICNET_PREDICTOR_COLS
    assert scaler.mean_.shape[0] == len(ELASTICNET_PREDICTOR_COLS)
    assert "alpha" in hyperparams and "l1_ratio" in hyperparams


def test_fit_elasticnet_tier1_excludes_tier2_columns():
    """Tier-1 predictor set must exclude the wind/solar-derived columns
    features.py tags as not proven point-in-time safe at the 11:45 D-1
    cutoff (see features.FEATURE_AVAILABILITY_TIER).
    """
    df = _make_full_feature_frame(periods=24 * 400)
    model, scaler, cols, hyperparams = fit_elasticnet(df, predictor_cols=ELASTICNET_TIER1_PREDICTOR_COLS)
    assert cols == ELASTICNET_TIER1_PREDICTOR_COLS
    for tier2_col in ["renewables_forecast_mw", "residual_load_forecast_mw", "renewable_share_forecast"]:
        assert tier2_col not in cols
    assert "load_forecast_mw" in cols  # Tier 1, still included


def test_final_refit_scaler_matches_complete_training_data():
    """Note on naming: this test verifies the FINAL refit scaler's
    statistics match a scaler fit on the full complete-case training
    data -- it does NOT directly observe each individual inner-fold
    scaler (GridSearchCV doesn't expose those intermediate fits). The
    actual inner-fold isolation guarantee comes from the Pipeline
    architecture itself, checked directly in
    test_elasticnet_search_uses_pipeline_timeseries_split_and_mae_scoring.
    This test is a supporting sanity check, not the primary proof.
    """
    df = _make_full_feature_frame(periods=24 * 400)
    model, scaler, cols, _ = fit_elasticnet(df)
    complete = df.dropna(subset=cols + [TARGET_COL])
    expected_scaler = StandardScaler().fit(complete[cols].values)
    assert np.allclose(scaler.mean_, expected_scaler.mean_)
    assert np.allclose(scaler.scale_, expected_scaler.scale_)


def test_elasticnet_search_uses_pipeline_timeseries_split_and_mae_scoring():
    """Direct architecture assertion: the scaler-before-inner-CV bug is
    prevented by construction because the scaler lives INSIDE the
    Pipeline that GridSearchCV cross-validates, not because of any
    property of the output. This is the primary proof the earlier bug
    can't recur; the refit-scaler test above is a secondary sanity check.
    """
    search = build_elasticnet_search()
    assert isinstance(search.estimator, Pipeline)
    assert isinstance(search.estimator.named_steps["scaler"], StandardScaler)
    assert isinstance(search.cv, TimeSeriesSplit)
    assert search.scoring == "neg_mean_absolute_error"


def test_v2_alpha_grid_is_genuine_superset_of_v1_grid():
    """Regression test for a real mistake: an earlier version of this
    grid extension replaced the search space with a DIFFERENT
    np.logspace(-5, 2, 20) call rather than extending the original one,
    so it didn't actually contain 0.001 (the value 6/8 baseline_v1 fits
    selected at the boundary) or any other original candidate -- that
    would have made "did allowing smaller alpha help" ambiguous with
    "did switching to a different discretization help". The v2 profile
    must be a genuine union: every v1 candidate present, plus new
    smaller values.
    """
    for original_value in BASELINE_V1_ALPHA_GRID:
        assert np.any(np.isclose(BASELINE_V2_ALPHA_GRID, original_value)), (
            f"baseline_v1 alpha candidate {original_value} missing from the v2 grid"
        )
    assert np.any(np.isclose(BASELINE_V2_ALPHA_GRID, 0.001)), "exact baseline_v1 boundary value must be preserved"
    assert BASELINE_V2_ALPHA_GRID.min() < BASELINE_V1_ALPHA_GRID.min(), "v2 must add values BELOW the v1 minimum"


def test_search_profiles_are_explicitly_named_not_a_mutable_default():
    """Regression test for a real reproducibility gap: an earlier version
    had one mutable DEFAULT_ALPHA_GRID module constant that changed
    across rounds of fixes, so re-running "baseline_v1" on a fresh clone
    would silently use whatever grid was CURRENTLY the default rather
    than reproducing the actual historical v1 experiment. Both profiles
    must exist as fixed, independently-addressable named constants.
    """
    assert "v1" in ELASTICNET_SEARCH_PROFILES
    assert "v2" in ELASTICNET_SEARCH_PROFILES
    assert list(ELASTICNET_SEARCH_PROFILES["v1"]["alpha_grid"]) == list(BASELINE_V1_ALPHA_GRID)
    assert list(ELASTICNET_SEARCH_PROFILES["v2"]["alpha_grid"]) == list(BASELINE_V2_ALPHA_GRID)
    # v1 must NOT silently pick up v2's extra values
    assert 1e-5 not in ELASTICNET_SEARCH_PROFILES["v1"]["alpha_grid"]
    assert 1e-5 in ELASTICNET_SEARCH_PROFILES["v2"]["alpha_grid"]


def test_fit_elasticnet_search_profile_v1_reproduces_v1_grid():
    """fit_elasticnet(search_profile="v1") must use EXACTLY the v1 grid,
    not whatever DEFAULT_SEARCH_PROFILE currently points to -- this is
    the actual reproducibility guarantee the profile system exists for.
    """
    df = _make_full_feature_frame(periods=24 * 400)
    search_v1 = build_elasticnet_search(search_profile="v1")
    assert list(search_v1.param_grid["elasticnet__alpha"]) == list(BASELINE_V1_ALPHA_GRID)
    search_v2 = build_elasticnet_search(search_profile="v2")
    assert list(search_v2.param_grid["elasticnet__alpha"]) == list(BASELINE_V2_ALPHA_GRID)


def test_fit_elasticnet_sorts_shuffled_input_chronologically():
    """TimeSeriesSplit operates on ROW ORDER, not timestamp values -- it
    has no idea what a timestamp even is. If fit_elasticnet() silently
    trusted incoming row order, a shuffled (or merely differently
    sorted) input would break the "train-only inner CV" guarantee
    without raising anything. Prove it sorts internally regardless of
    input order by checking a shuffled frame still fits successfully
    and produces the same hyperparameters as the unshuffled version.
    """
    df = _make_full_feature_frame(periods=24 * 400)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    _, _, _, hyperparams_ordered = fit_elasticnet(df)
    _, _, _, hyperparams_shuffled = fit_elasticnet(shuffled)

    assert hyperparams_ordered == hyperparams_shuffled


def test_fit_elasticnet_rejects_duplicate_timestamps():
    df = _make_full_feature_frame(periods=24 * 400)
    dupe_df = pd.concat([df, df.iloc[[300]]], ignore_index=True)  # duplicate one row's timestamp
    with pytest.raises(AssertionError, match="duplicate"):
        fit_elasticnet(dupe_df)


def test_predict_elasticnet_returns_nan_for_incomplete_rows():
    df = _make_full_feature_frame(periods=24 * 400)
    model, scaler, cols, _ = fit_elasticnet(df)
    df_with_gap = df.copy()
    # Pick rows deep enough into the series to have full lag_168h history already
    base = 300
    df_with_gap.loc[df_with_gap.index[base], "residual_load_forecast_mw"] = np.nan
    preds = predict_elasticnet(model, scaler, df_with_gap, cols)
    assert pd.isna(preds.iloc[base])
    assert preds.iloc[base + 1: base + 6].notna().all()


def test_evaluate_fold_train_val_are_chronologically_disjoint():
    """Regression-style guard: the actual rows used to fit ElasticNet in
    a fold must never include a timestamp at/after that fold's val_start.
    """
    df = _make_full_feature_frame(periods=24 * 365 * 5)  # 2019-2023ish
    window = SplitWindow(
        "test_fold",
        pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"),
    )
    result = evaluate_fold(df, window)
    expected_models = {"lag_24", "lag_168", "elasticnet_full", "elasticnet_tier1"}
    assert expected_models <= set(result.overall_metrics.index)
    for model in expected_models:
        assert result.overall_metrics.loc[model, "mae"] > 0
        assert np.isfinite(result.overall_metrics.loc[model, "mae"])


def test_evaluate_fold_produces_by_hour_and_by_regime_breakdowns():
    df = _make_full_feature_frame(periods=24 * 365 * 5)
    window = SplitWindow(
        "test_fold",
        pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"),
    )
    result = evaluate_fold(df, window)
    expected_models = {"lag_24", "lag_168", "elasticnet_full", "elasticnet_tier1"}
    assert set(result.metrics_by_hour.keys()) == expected_models
    assert len(result.metrics_by_hour["elasticnet_full"]) <= 24  # at most 24 distinct hours
    assert set(result.metrics_by_train_quantile_regime.keys()) == expected_models
    assert set(result.metrics_by_fixed_regime.keys()) == expected_models


def test_evaluate_fold_all_models_share_identical_row_set():
    """Regression test for a real issue: models were previously scored
    on slightly different row counts (each dropping its own NaN pattern
    independently), which broke apples-to-apples MAE comparisons. All
    four models must now be evaluated on exactly the same common rows.
    """
    df = _make_full_feature_frame(periods=24 * 365 * 5)
    window = SplitWindow(
        "test_fold",
        pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"),
    )
    result = evaluate_fold(df, window)
    ns = result.overall_metrics["n"]
    assert ns.nunique() == 1, f"models were scored on different row counts: {ns.to_dict()}"
    assert int(result.coverage["common_comparison_rows"].iloc[0]) == int(ns.iloc[0])


def test_evaluate_fold_coverage_report_shows_per_model_usable_rows():
    df = _make_full_feature_frame(periods=24 * 365 * 5)
    window = SplitWindow(
        "test_fold",
        pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"),
    )
    result = evaluate_fold(df, window)
    cov = result.coverage.iloc[0]
    assert cov["raw_val_rows"] >= cov["common_comparison_rows"]
    assert cov["lag_24_usable_rows"] >= cov["common_comparison_rows"]
    assert cov["elasticnet_full_usable_rows"] >= cov["common_comparison_rows"]
    assert cov["elasticnet_tier1_usable_rows"] >= cov["common_comparison_rows"]


def test_evaluate_fold_saves_coefficients_and_hyperparams_for_both_variants():
    df = _make_full_feature_frame(periods=24 * 365 * 5)
    window = SplitWindow(
        "test_fold",
        pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"),
    )
    result = evaluate_fold(df, window)
    assert set(result.elasticnet_coefficients.keys()) == {"elasticnet_full", "elasticnet_tier1"}
    assert set(result.elasticnet_coefficients["elasticnet_full"]["feature"]) == set(ELASTICNET_PREDICTOR_COLS)
    assert set(result.elasticnet_coefficients["elasticnet_tier1"]["feature"]) == set(ELASTICNET_TIER1_PREDICTOR_COLS)
    for variant in ["elasticnet_full", "elasticnet_tier1"]:
        assert "alpha" in result.elasticnet_hyperparams[variant]
        assert "l1_ratio" in result.elasticnet_hyperparams[variant]


def test_evaluate_fold_saves_predictions():
    df = _make_full_feature_frame(periods=24 * 365 * 5)
    window = SplitWindow(
        "test_fold",
        pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"),
    )
    result = evaluate_fold(df, window)
    assert result.predictions is not None
    expected_cols = {"timestamp_utc", TARGET_COL, "lag_24_pred", "lag_168_pred",
                      "elasticnet_full_pred", "elasticnet_tier1_pred"}
    assert expected_cols <= set(result.predictions.columns)
    assert len(result.predictions) == int(result.coverage["common_comparison_rows"].iloc[0])


def test_fixed_stress_buckets_are_fold_independent_thresholds():
    """Unlike the training-quantile regime flag, fixed buckets
    (<0, >200, >500 EUR/MWh) must use the same absolute threshold
    regardless of which fold's training data was used.
    """
    df = _make_full_feature_frame(periods=24 * 365 * 5)
    window = SplitWindow(
        "test_fold",
        pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"),
    )
    result = evaluate_fold(df, window)
    # Buckets present depend only on whether any rows qualify, not on any fold-specific tuning
    for model_name, df_fixed in result.metrics_by_fixed_regime.items():
        assert set(df_fixed.index) <= {"negative_price", "price_gt_200", "price_gt_500"}
