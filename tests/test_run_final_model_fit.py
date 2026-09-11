"""Tests for run_final_model_fit.py.

The final-fit runner is implementation-only and must fail closed if any
part of the frozen Section 3 contract drifts:
  - Europe/Berlin final-train/holdout boundary,
  - 36-combination XGBoost grid,
  - MAE scoring,
  - fixed seed/objective/tree method,
  - frozen Full/Tier-1 predictor-set routing,
  - independent selection,
  - output overwrite,
  - exact freeze lineage in the saved manifest.
"""
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_final_model_fit as rfmf
from src.clean import local_delivery_date_to_utc


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# CLI: exactly two arguments, no override mechanism.
# ---------------------------------------------------------------------
def test_resolve_run_args_requires_exactly_two():
    with pytest.raises(SystemExit):
        rfmf.resolve_run_args([])
    with pytest.raises(SystemExit):
        rfmf.resolve_run_args(["delu_features.parquet"])
    with pytest.raises(SystemExit):
        rfmf.resolve_run_args(
            ["delu_features.parquet", "final_model_fit_v1", "extra_arg"]
        )


def test_resolve_run_args_valid_call():
    assert rfmf.resolve_run_args(
        ["delu_features.parquet", "final_model_fit_v1"]
    ) == ("delu_features.parquet", "final_model_fit_v1")


def test_no_cli_override_surface():
    sig = inspect.signature(rfmf.resolve_run_args)
    assert list(sig.parameters) == ["args"]


# ---------------------------------------------------------------------
# Frozen scientific contract.
# ---------------------------------------------------------------------
def test_frozen_constants_match_protocol():
    assert rfmf.RANDOM_SEED == 42
    assert rfmf.INNER_CV_SPLITS == 3
    assert rfmf.SCORING == "neg_mean_absolute_error"
    assert rfmf.PROTOCOL_TAG == "pre-holdout-protocol-v2"
    assert (
        rfmf.PROTOCOL_COMMIT
        == "a2d1f9a79022ca1e2731d52803061825038c69f5"
    )
    assert rfmf.DEVELOPMENT_TAG == "corrected-development-v2-dayorigin"
    assert (
        rfmf.DEVELOPMENT_COMMIT
        == "704cc933b99bb32c21cd358e613d1cd667d71488"
    )
    assert rfmf.XGBOOST_DEVELOPMENT_RUN == "xgboost_v1_a03fix"
    assert rfmf.FROZEN_TARGET_COL == "price_eur_mwh"


def test_frozen_grid_exact_values_and_size():
    assert rfmf.FROZEN_XGBOOST_PARAM_GRID == {
        "max_depth": [3, 4, 5],
        "n_estimators": [200, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    from sklearn.model_selection import ParameterGrid

    assert len(list(ParameterGrid(rfmf.FROZEN_XGBOOST_PARAM_GRID))) == 36


def test_assert_frozen_search_contract_passes_current_implementation():
    rfmf.assert_frozen_search_contract()


def test_assert_frozen_search_contract_rejects_scoring_drift(monkeypatch):
    fake_estimator = SimpleNamespace(
        get_params=lambda: {
            "random_state": 42,
            "objective": "reg:absoluteerror",
            "tree_method": "hist",
        }
    )
    fake_search = SimpleNamespace(
        scoring="neg_root_mean_squared_error",
        estimator=fake_estimator,
    )
    monkeypatch.setattr(
        rfmf,
        "build_xgboost_search",
        lambda **kwargs: fake_search,
    )

    with pytest.raises(AssertionError, match="scoring drifted"):
        rfmf.assert_frozen_search_contract()


def test_assert_frozen_search_contract_rejects_seed_drift(monkeypatch):
    fake_estimator = SimpleNamespace(
        get_params=lambda: {
            "random_state": 7,
            "objective": "reg:absoluteerror",
            "tree_method": "hist",
        }
    )
    fake_search = SimpleNamespace(
        scoring="neg_mean_absolute_error",
        estimator=fake_estimator,
    )
    monkeypatch.setattr(
        rfmf,
        "build_xgboost_search",
        lambda **kwargs: fake_search,
    )

    with pytest.raises(AssertionError, match="random seed drifted"):
        rfmf.assert_frozen_search_contract()


def test_fit_final_model_has_no_predictor_list_parameter():
    sig = inspect.signature(rfmf.fit_final_model)
    assert list(sig.parameters) == ["train_df", "label"]


def test_fit_final_model_rejects_unknown_information_set():
    with pytest.raises(ValueError, match="Unknown frozen information set"):
        rfmf.fit_final_model(pd.DataFrame(), "alternative_full")


def test_frozen_predictor_sets_are_exact_and_contract_passes():
    expected_full = (
        "load_forecast_mw",
        "renewables_forecast_mw",
        "residual_load_forecast_mw",
        "renewable_share_forecast",
        "price_lag_24h",
        "price_lag_48h",
        "price_lag_168h",
        "price_rolling_mean_24h",
        "price_rolling_vol_24h",
        "price_rolling_mean_168h",
        "price_rolling_vol_168h",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "weekend",
        "weekend_hour_sin",
        "weekend_hour_cos",
        "post_15min_mtu",
    )
    expected_tier1 = (
        "load_forecast_mw",
        "price_lag_24h",
        "price_lag_48h",
        "price_lag_168h",
        "price_rolling_mean_24h",
        "price_rolling_vol_24h",
        "price_rolling_mean_168h",
        "price_rolling_vol_168h",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "weekend",
        "weekend_hour_sin",
        "weekend_hour_cos",
        "post_15min_mtu",
    )

    assert rfmf.FROZEN_FULL_PREDICTOR_COLS == expected_full
    assert rfmf.FROZEN_TIER1_PREDICTOR_COLS == expected_tier1
    assert set(expected_tier1) < set(expected_full)
    rfmf.assert_frozen_predictor_contract()


def test_frozen_predictor_contract_rejects_full_drift(monkeypatch):
    monkeypatch.setattr(
        rfmf,
        "XGBOOST_PREDICTOR_COLS",
        list(rfmf.FROZEN_FULL_PREDICTOR_COLS) + ["post_result_feature"],
    )
    with pytest.raises(AssertionError, match="Full information set"):
        rfmf.assert_frozen_predictor_contract()


def test_frozen_predictor_contract_rejects_target_drift(monkeypatch):
    monkeypatch.setattr(rfmf, "TARGET_COL", "different_target")
    with pytest.raises(AssertionError, match="target column drifted"):
        rfmf.assert_frozen_predictor_contract()


# ---------------------------------------------------------------------
# Exact market-date boundary: 2026-01-01 Berlin == 2025-12-31 23:00 UTC.
# ---------------------------------------------------------------------
def test_holdout_boundary_is_berlin_midnight_not_utc_midnight():
    boundary = pd.Timestamp(local_delivery_date_to_utc("2026-01-01"))
    assert boundary == pd.Timestamp("2025-12-31T23:00:00Z")


def test_frozen_train_window_contract_passes_current_split_definition():
    from src.splits import get_final_train_window

    rfmf.assert_frozen_train_window(get_final_train_window())


def test_frozen_train_window_contract_rejects_early_end():
    from src.splits import SplitWindow

    start = pd.Timestamp(local_delivery_date_to_utc("2019-01-01"))
    wrong_end = pd.Timestamp(local_delivery_date_to_utc("2025-12-31"))
    wrong = SplitWindow(
        "final_train",
        start,
        wrong_end,
        wrong_end,
        wrong_end,
    )

    with pytest.raises(AssertionError, match="end drifted"):
        rfmf.assert_frozen_train_window(wrong)


def test_final_train_window_excludes_exact_berlin_boundary():
    from src.splits import get_final_train_window, slice_window

    boundary = pd.Timestamp(local_delivery_date_to_utc("2026-01-01"))
    df = pd.DataFrame(
        {
            "timestamp_utc": [
                boundary - pd.Timedelta(hours=1),
                boundary,
                boundary + pd.Timedelta(hours=1),
            ]
        }
    )
    train, _ = slice_window(df, get_final_train_window())

    assert train["timestamp_utc"].tolist() == [
        boundary - pd.Timedelta(hours=1)
    ]


def test_full_and_tier1_fits_route_independently_to_exact_predictor_sets(monkeypatch):
    calls = []

    class DummyModel:
        pass

    def fake_fit_xgboost(
        train_df,
        predictor_cols,
        target_col,
        inner_cv_splits,
        param_grid,
        day_aligned_cv,
    ):
        calls.append(
            {
                "predictor_cols": tuple(predictor_cols),
                "target_col": target_col,
                "inner_cv_splits": inner_cv_splits,
                "param_grid": param_grid,
                "day_aligned_cv": day_aligned_cv,
            }
        )
        return DummyModel(), predictor_cols, {"max_depth": 3}

    monkeypatch.setattr(rfmf, "fit_xgboost", fake_fit_xgboost)

    row = {rfmf.FROZEN_TARGET_COL: 50.0}
    for col in rfmf.FROZEN_FULL_PREDICTOR_COLS:
        row[col] = 1.0
    df = pd.DataFrame([row])

    full = rfmf.fit_final_model(df, "full")
    tier1 = rfmf.fit_final_model(df, "tier1")

    assert len(calls) == 2
    assert calls[0]["predictor_cols"] == rfmf.FROZEN_FULL_PREDICTOR_COLS
    assert calls[1]["predictor_cols"] == rfmf.FROZEN_TIER1_PREDICTOR_COLS
    assert calls[0]["target_col"] == "price_eur_mwh"
    assert calls[1]["target_col"] == "price_eur_mwh"
    assert calls[0]["inner_cv_splits"] == 3
    assert calls[1]["inner_cv_splits"] == 3
    assert calls[0]["day_aligned_cv"] is True
    assert calls[1]["day_aligned_cv"] is True
    assert calls[0]["param_grid"] == rfmf.FROZEN_XGBOOST_PARAM_GRID
    assert calls[1]["param_grid"] == rfmf.FROZEN_XGBOOST_PARAM_GRID
    assert full["hyperparameters"] is not tier1["hyperparameters"]


# ---------------------------------------------------------------------
# Synthetic end-to-end fixtures.
# ---------------------------------------------------------------------
def _build_synthetic_features(n_days=60):
    from src.models import ELASTICNET_PREDICTOR_COLS, TARGET_COL

    rng = np.random.default_rng(0)
    ts = pd.date_range(
        "2025-12-20",
        periods=n_days * 24,
        freq="1h",
        tz="UTC",
    )
    rows = []
    for t in ts:
        row = {
            "timestamp_utc": t,
            TARGET_COL: 50 + rng.normal(0, 10),
        }
        for col in ELASTICNET_PREDICTOR_COLS:
            row[col] = rng.normal(0, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def test_full_end_to_end_run(tmp_path, monkeypatch):
    import run_final_model_fit as m

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    df = _build_synthetic_features()
    holdout_start_utc = pd.Timestamp(
        local_delivery_date_to_utc("2026-01-01")
    )
    n_holdout_rows = int(
        (df["timestamp_utc"] >= holdout_start_utc).sum()
    )
    assert n_holdout_rows > 0

    df.to_parquet(processed_dir / "test_features.parquet")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_model_fit.py",
            "test_features.parquet",
            "final_model_fit_v1",
        ],
    )
    m.main()

    out_dir = (
        tmp_path
        / "outputs"
        / "final_model_fit"
        / "test_features"
        / "final_model_fit_v1"
    )

    for filename in (
        "full_model.json",
        "tier1_model.json",
        "full_hyperparameters.json",
        "tier1_hyperparameters.json",
        "full_predictor_cols.json",
        "tier1_predictor_cols.json",
        "final_model_fit_manifest.json",
    ):
        assert (out_dir / filename).exists()

    manifest = json.loads(
        (out_dir / "final_model_fit_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["holdout_used"] is False
    assert manifest["random_seed"] == 42
    assert manifest["day_aligned_cv"] is True
    assert manifest["scoring"] == "neg_mean_absolute_error"
    assert manifest["param_grid_n_combinations"] == 36
    assert manifest["target_col"] == "price_eur_mwh"
    assert (
        manifest["selection_independent_per_information_set"] is True
    )

    assert (
        manifest["protocol_tag"]
        == "pre-holdout-protocol-v2"
    )
    assert (
        manifest["protocol_commit"]
        == "a2d1f9a79022ca1e2731d52803061825038c69f5"
    )
    assert (
        manifest["development_evidence_tag"]
        == "corrected-development-v2-dayorigin"
    )
    assert (
        manifest["development_evidence_commit"]
        == "704cc933b99bb32c21cd358e613d1cd667d71488"
    )
    assert manifest["xgboost_development_run"] == "xgboost_v1_a03fix"
    assert tuple(manifest["frozen_full_predictor_cols"]) == rfmf.FROZEN_FULL_PREDICTOR_COLS
    assert tuple(manifest["frozen_tier1_predictor_cols"]) == rfmf.FROZEN_TIER1_PREDICTOR_COLS

    assert (
        manifest["full_n_training_rows"]
        < len(df)
    )

    full_predictors = json.loads(
        (out_dir / "full_predictor_cols.json").read_text(
            encoding="utf-8"
        )
    )
    tier1_predictors = json.loads(
        (out_dir / "tier1_predictor_cols.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(tier1_predictors) < set(full_predictors)


def test_output_overwrite_rejected(tmp_path, monkeypatch):
    import run_final_model_fit as m

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    _build_synthetic_features().to_parquet(
        processed_dir / "test_features.parquet"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_model_fit.py",
            "test_features.parquet",
            "final_model_fit_v1",
        ],
    )

    m.main()
    with pytest.raises(FileExistsError):
        m.main()


def test_missing_input_file_raises_clearly(tmp_path, monkeypatch):
    import run_final_model_fit as m

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_model_fit.py",
            "never_created.parquet",
            "v1",
        ],
    )

    with pytest.raises(FileNotFoundError, match="Missing"):
        m.main()


def test_2026_rows_genuinely_excluded_from_training_row_count(
    tmp_path,
    monkeypatch,
):
    import run_final_model_fit as m
    from src.models import ELASTICNET_PREDICTOR_COLS, TARGET_COL

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    rng = np.random.default_rng(1)
    holdout_start_utc = pd.Timestamp(
        local_delivery_date_to_utc("2026-01-01")
    )

    # Exactly 20 days strictly before the Berlin boundary and 20 days
    # starting exactly at the boundary.
    pre_ts = pd.date_range(
        end=holdout_start_utc - pd.Timedelta(hours=1),
        periods=20 * 24,
        freq="1h",
    )
    post_ts = pd.date_range(
        start=holdout_start_utc,
        periods=20 * 24,
        freq="1h",
    )

    rows = []
    for ts_range in (pre_ts, post_ts):
        for t in ts_range:
            row = {
                "timestamp_utc": t,
                TARGET_COL: 50 + rng.normal(0, 10),
            }
            for col in ELASTICNET_PREDICTOR_COLS:
                row[col] = rng.normal(0, 1)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(processed_dir / "test_features.parquet")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_model_fit.py",
            "test_features.parquet",
            "v1",
        ],
    )
    m.main()

    out_dir = (
        tmp_path
        / "outputs"
        / "final_model_fit"
        / "test_features"
        / "v1"
    )
    manifest = json.loads(
        (out_dir / "final_model_fit_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    n_pre_2026 = 20 * 24
    assert manifest["raw_rows_in_final_train_window"] == n_pre_2026
    assert manifest["full_n_training_rows"] <= n_pre_2026
    assert manifest["tier1_n_training_rows"] <= n_pre_2026