"""Tests for the XGBoost challenger model. Mirrors test_models.py's
patterns: chronological ordering enforced (not assumed), predictor set
correctness, common-mask evaluation across all six models, and the
same class of regression tests that caught real issues in the
ElasticNet stage.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.xgboost_model import (
    XGBOOST_PREDICTOR_COLS,
    XGBOOST_TIER1_PREDICTOR_COLS,
    XGBOOST_PARAM_GRID,
    build_xgboost_search,
    fit_xgboost,
    predict_xgboost,
    evaluate_fold_with_xgboost,
    make_delivery_day_cv,
)
from src.models import ELASTICNET_PREDICTOR_COLS, ELASTICNET_TIER1_PREDICTOR_COLS, TARGET_COL
from src.splits import SplitWindow
from src.clean import local_delivery_date_to_utc

# Tiny grid + fewer inner splits for test speed -- NOT the production
# grid (see test_xgboost_param_grid_matches_config_yaml for that).
_FAST_GRID = {"max_depth": [3], "n_estimators": [30], "learning_rate": [0.1], "subsample": [1.0]}


def _make_full_feature_frame(periods=24 * 365 * 5, seed=0):
    """Same synthetic-data generator shape as test_models.py, so the
    two model families are exercised identically.
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


def test_xgboost_predictor_sets_are_identical_to_elasticnets():
    """Regression guard for the module's central design decision: any
    MAE difference between XGBoost and ElasticNet must reflect the
    modelling approach, not a different information set. If someone
    later adds an XGBoost-only feature (e.g. raw hour_local, since trees
    don't need cyclic encoding), that's a deliberate, separate decision
    -- not something that should happen by accident.
    """
    assert XGBOOST_PREDICTOR_COLS == ELASTICNET_PREDICTOR_COLS
    assert XGBOOST_TIER1_PREDICTOR_COLS == ELASTICNET_TIER1_PREDICTOR_COLS


def test_xgboost_param_grid_matches_config_yaml():
    """The grid must match what was frozen in config.yaml BEFORE any
    model was built (spec section 9: 'tune only a small parameter
    space'), not an ad-hoc grid invented after seeing results. This
    reads config.yaml directly rather than comparing against a second
    hardcoded dict, which would silently drift out of sync if
    config.yaml were ever edited without a corresponding code change.
    """
    from src.utils import load_config
    cfg = load_config()
    expected = cfg["models"]["xgboost_param_grid"]
    assert XGBOOST_PARAM_GRID == expected


def test_xgboost_search_uses_mae_scoring_and_timeseries_split():
    search = build_xgboost_search()
    assert search.scoring == "neg_mean_absolute_error"
    from sklearn.model_selection import TimeSeriesSplit
    assert isinstance(search.cv, TimeSeriesSplit)


def test_delivery_day_cv_no_day_appears_in_both_train_and_val():
    """Core correctness guarantee: a Europe/Berlin delivery date must
    never be split across the inner train/val boundary -- all of its
    hours go to exactly one side.
    """
    ts = pd.date_range("2023-01-01T00:00:00Z", periods=24 * 60, freq="1h", tz="UTC")
    splits = make_delivery_day_cv(pd.Series(ts), n_splits=3)
    local_days = pd.to_datetime(ts).tz_convert("Europe/Berlin").date
    for train_idx, val_idx in splits:
        train_days = set(local_days[train_idx])
        val_days = set(local_days[val_idx])
        assert train_days.isdisjoint(val_days), "a delivery date appears on both sides of an inner split"


def test_fit_xgboost_uses_day_aligned_cv_by_default():
    """Freeze-quality regression guard: a future edit could silently
    flip fit_xgboost's day_aligned_cv default from True to False while
    every standalone make_delivery_day_cv() test kept passing (those
    only test the splitter function in isolation, not that fit_xgboost
    actually calls it by default). This intercepts the actual `cv`
    argument passed to build_xgboost_search() during a real fit_xgboost()
    call and checks it's a list of (train_idx, val_idx) tuples (the
    day-aligned path), not a bare TimeSeriesSplit object (the old,
    hourly-row path).
    """
    from unittest.mock import patch
    import src.xgboost_model as xm

    df = _make_full_feature_frame(periods=24 * 200)
    original_build_search = xm.build_xgboost_search
    captured = {}

    def spy_build_search(*args, **kwargs):
        captured["cv"] = kwargs.get("cv")
        return original_build_search(*args, **kwargs)

    with patch.object(xm, "build_xgboost_search", side_effect=spy_build_search):
        xm.fit_xgboost(df, param_grid=_FAST_GRID, inner_cv_splits=2)  # day_aligned_cv defaults to True

    assert captured["cv"] is not None, "fit_xgboost's default did not pass a day-aligned cv to build_xgboost_search"
    assert isinstance(captured["cv"], list), "expected a list of (train_idx, val_idx) tuples from make_delivery_day_cv"


def test_delivery_day_cv_train_precedes_val_chronologically():
    ts = pd.date_range("2023-01-01T00:00:00Z", periods=24 * 60, freq="1h", tz="UTC")
    splits = make_delivery_day_cv(pd.Series(ts), n_splits=3)
    for train_idx, val_idx in splits:
        assert ts[train_idx].max() < ts[val_idx].min()


def test_delivery_day_cv_keeps_spring_dst_day_whole():
    """2024-03-31 is Germany's spring-forward DST transition -- a
    23-local-hour day. It must stay entirely on one side of every
    inner split, never split mid-day.
    """
    ts = pd.date_range("2024-01-01T00:00:00Z", periods=24 * 150, freq="1h", tz="UTC")
    splits = make_delivery_day_cv(pd.Series(ts), n_splits=3)
    local_days = pd.to_datetime(ts).tz_convert("Europe/Berlin").date
    spring_dst_date = pd.Timestamp("2024-03-31").date()
    for train_idx, val_idx in splits:
        rows_on_that_day_in_train = (local_days[train_idx] == spring_dst_date).sum()
        rows_on_that_day_in_val = (local_days[val_idx] == spring_dst_date).sum()
        assert rows_on_that_day_in_train == 0 or rows_on_that_day_in_val == 0, (
            "spring DST day was split across train/val"
        )


def test_delivery_day_cv_keeps_autumn_dst_day_whole():
    """2024-10-27 is Germany's fall-back DST transition -- a 25-local-hour
    day. Same guarantee as the spring case.
    """
    ts = pd.date_range("2024-08-01T00:00:00Z", periods=24 * 150, freq="1h", tz="UTC")
    splits = make_delivery_day_cv(pd.Series(ts), n_splits=3)
    local_days = pd.to_datetime(ts).tz_convert("Europe/Berlin").date
    autumn_dst_date = pd.Timestamp("2024-10-27").date()
    for train_idx, val_idx in splits:
        rows_on_that_day_in_train = (local_days[train_idx] == autumn_dst_date).sum()
        rows_on_that_day_in_val = (local_days[val_idx] == autumn_dst_date).sum()
        assert rows_on_that_day_in_train == 0 or rows_on_that_day_in_val == 0, (
            "autumn DST day was split across train/val"
        )


def test_fit_xgboost_rejects_duplicate_timestamps():
    df = _make_full_feature_frame(periods=24 * 200)
    dupe_df = pd.concat([df, df.iloc[[300]]], ignore_index=True)  # row 300 has full lag_168h history
    with pytest.raises(AssertionError, match="duplicate"):
        fit_xgboost(dupe_df, param_grid=_FAST_GRID, inner_cv_splits=2)


def test_fit_xgboost_sorts_shuffled_input_chronologically():
    df = _make_full_feature_frame(periods=24 * 200)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    _, _, hyperparams_ordered = fit_xgboost(df, param_grid=_FAST_GRID, inner_cv_splits=2)
    _, _, hyperparams_shuffled = fit_xgboost(shuffled, param_grid=_FAST_GRID, inner_cv_splits=2)
    assert hyperparams_ordered == hyperparams_shuffled


def test_predict_xgboost_returns_nan_for_incomplete_rows():
    df = _make_full_feature_frame(periods=24 * 200)
    model, cols, _ = fit_xgboost(df, param_grid=_FAST_GRID, inner_cv_splits=2)
    df_with_gap = df.copy()
    base = 300  # deep enough to have full price_lag_168h history already
    df_with_gap.loc[df_with_gap.index[base], "residual_load_forecast_mw"] = np.nan
    preds = predict_xgboost(model, df_with_gap, cols)
    assert pd.isna(preds.iloc[base])
    assert preds.iloc[base + 1: base + 6].notna().all()


def test_evaluate_fold_with_xgboost_all_six_models_share_identical_row_set():
    """The core correctness guarantee, extended from models.py's
    four-model version to all six models now that XGBoost is included.
    """
    df = _make_full_feature_frame(periods=24 * 365 * 4)
    window = SplitWindow(
        "test_fold",
        pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"),
    )
    result = evaluate_fold_with_xgboost(df, window, xgboost_param_grid=_FAST_GRID, xgboost_inner_cv_splits=2)
    ns = result.overall_metrics["n"]
    assert ns.nunique() == 1, f"models were scored on different row counts: {ns.to_dict()}"
    expected_models = {"lag_24", "lag_168", "elasticnet_full", "elasticnet_tier1", "xgboost_full", "xgboost_tier1"}
    assert set(result.overall_metrics.index) == expected_models
    assert int(result.coverage["common_comparison_rows"].iloc[0]) == int(ns.iloc[0])


def test_evaluate_fold_with_xgboost_saves_feature_importances():
    df = _make_full_feature_frame(periods=24 * 365 * 4)
    window = SplitWindow(
        "test_fold",
        pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"),
    )
    result = evaluate_fold_with_xgboost(df, window, xgboost_param_grid=_FAST_GRID, xgboost_inner_cv_splits=2)
    assert set(result.xgboost_feature_importances.keys()) == {"xgboost_full", "xgboost_tier1"}
    full_importances = result.xgboost_feature_importances["xgboost_full"]
    assert set(full_importances["feature"]) == set(XGBOOST_PREDICTOR_COLS)
    # importances should be non-negative and sum to ~1 (XGBoost's default gain-based importance)
    assert (full_importances["importance"] >= 0).all()


def test_evaluate_fold_with_xgboost_uses_frozen_elasticnet_v1_profile_by_default():
    """XGBoost's comparison must use the ACCEPTED baseline_v1 ElasticNet
    profile by default, not silently drift to v2 or some other grid --
    baseline_v1 was frozen specifically so later stages have a stable
    reference point.
    """
    df = _make_full_feature_frame(periods=24 * 365 * 4)
    window = SplitWindow(
        "test_fold",
        pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-01", tz="UTC"),
    )
    result = evaluate_fold_with_xgboost(df, window, xgboost_param_grid=_FAST_GRID, xgboost_inner_cv_splits=2)
    from src.models import BASELINE_V1_ALPHA_GRID
    full_alpha = result.elasticnet_hyperparams["elasticnet_full"]["alpha"]
    assert any(np.isclose(full_alpha, v) for v in BASELINE_V1_ALPHA_GRID), (
        f"alpha={full_alpha} not in the v1 grid -- default profile may have drifted"
    )
