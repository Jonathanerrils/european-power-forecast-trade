"""Tests for run_final_holdout.py.

These tests exercise the pre-registered information-set and provenance
properties without running the real 2026 holdout.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_final_holdout as rfh
from src.clean import local_delivery_date_to_utc


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest().upper()


def test_resolve_run_args_accepts_only_output_version():
    assert rfh.resolve_run_args(["holdout_v1"]) == "holdout_v1"
    with pytest.raises(SystemExit):
        rfh.resolve_run_args([])
    with pytest.raises(SystemExit):
        rfh.resolve_run_args(
            ["delu_features.parquet", "xgboost_v1_a03fix", "holdout_v1"]
        )


def test_frozen_constants_match_protocol():
    assert rfh.INPUT_FILENAME == "delu_features.parquet"
    assert rfh.XGBOOST_RUN_VERSION == "xgboost_v1_a03fix"
    assert (
        rfh.UNCERTAINTY_SELECTED_RUN_VERSION
        == "uncertainty_selected_v2_dayorigin"
    )
    assert (
        rfh.TIER1_UNCERTAINTY_RUN_VERSION
        == "uncertainty_tier1_robustness_v3_dayorigin"
    )
    assert rfh.FINAL_MODEL_FIT_RUN_VERSION == "final_model_fit_v1"
    assert rfh.HOLDOUT_START_LOCAL == "2026-01-01"
    assert rfh.HOLDOUT_END_LOCAL_EXCLUSIVE == "2026-08-01"
    assert rfh.WINDOW_DAYS == 60
    assert rfh.MIN_PERIODS_DAYS == 15
    assert rfh.QUANTILES == [0.1, 0.5, 0.9]
    assert rfh.INFORMATION_SET_METHOD == "delivery_day_safe"
    assert (
        rfh.FINAL_MODEL_FIT_COMMIT
        == "54a7c2e77122134eca31a09798cc1614df1c634c"
    )


def test_frozen_economic_contract_passes():
    rfh.assert_frozen_economic_contract()


def _write_dummy_final_fit(tmp_path, monkeypatch, bad_protocol=False):
    from xgboost import XGBRegressor

    monkeypatch.setattr(rfh, "REPO_ROOT", tmp_path)
    fit_dir = (
        tmp_path
        / "outputs"
        / "final_model_fit"
        / rfh.INPUT_STEM
        / rfh.FINAL_MODEL_FIT_RUN_VERSION
    )
    fit_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    full_cols = ["x1", "x2"]
    tier1_cols = ["x1"]

    full_model = XGBRegressor(n_estimators=3, max_depth=2)
    full_model.fit(rng.normal(size=(50, 2)), rng.normal(size=50))
    full_model.save_model(str(fit_dir / "full_model.json"))

    tier1_model = XGBRegressor(n_estimators=3, max_depth=2)
    tier1_model.fit(rng.normal(size=(50, 1)), rng.normal(size=50))
    tier1_model.save_model(str(fit_dir / "tier1_model.json"))

    (fit_dir / "full_predictor_cols.json").write_text(
        json.dumps(full_cols), encoding="utf-8"
    )
    (fit_dir / "tier1_predictor_cols.json").write_text(
        json.dumps(tier1_cols), encoding="utf-8"
    )
    (fit_dir / "full_hyperparameters.json").write_text(
        json.dumps({"max_depth": 2}), encoding="utf-8"
    )
    (fit_dir / "tier1_hyperparameters.json").write_text(
        json.dumps({"max_depth": 2}), encoding="utf-8"
    )

    manifest = {
        "input_filename": rfh.INPUT_FILENAME,
        "output_run_version": rfh.FINAL_MODEL_FIT_RUN_VERSION,
        "train_start_local_date": "2019-01-01",
        "train_end_local_date_exclusive": rfh.HOLDOUT_START_LOCAL,
        "target_col": "price_eur_mwh",
        "param_grid_n_combinations": 36,
        "inner_cv_splits": 3,
        "day_aligned_cv": True,
        "scoring": "neg_mean_absolute_error",
        "random_seed": 42,
        "xgboost_objective": "reg:absoluteerror",
        "xgboost_tree_method": "hist",
        "selection_independent_per_information_set": True,
        "holdout_used": False,
        "protocol_tag": rfh.PROTOCOL_TAG,
        "protocol_commit": (
            "wrong" if bad_protocol else rfh.PROTOCOL_COMMIT
        ),
        "development_evidence_tag": rfh.DEVELOPMENT_TAG,
        "development_evidence_commit": rfh.DEVELOPMENT_COMMIT,
        "xgboost_development_run": rfh.XGBOOST_RUN_VERSION,
        "full_predictor_cols": full_cols,
        "frozen_full_predictor_cols": full_cols,
        "tier1_predictor_cols": tier1_cols,
        "frozen_tier1_predictor_cols": tier1_cols,
    }
    (fit_dir / "final_model_fit_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    expected = {}
    for filename in (
        "final_model_fit_manifest.json",
        "full_hyperparameters.json",
        "full_model.json",
        "full_predictor_cols.json",
        "tier1_hyperparameters.json",
        "tier1_model.json",
        "tier1_predictor_cols.json",
    ):
        expected[filename] = _sha(fit_dir / filename)

    monkeypatch.setattr(rfh, "EXPECTED_FINAL_FIT_SHA256", expected)
    (fit_dir / "SHA256SUMS.txt").write_text(
        "\n".join(
            f"{digest}  {filename}"
            for filename, digest in expected.items()
        ),
        encoding="utf-8",
    )
    return fit_dir


def test_load_final_models_accepts_exact_dummy_artifacts(
    tmp_path, monkeypatch
):
    _write_dummy_final_fit(tmp_path, monkeypatch)
    loaded = rfh.load_final_models()
    assert loaded["models"]["full"]["predictor_cols"] == ["x1", "x2"]
    assert loaded["models"]["tier1"]["predictor_cols"] == ["x1"]


def test_load_final_models_rejects_byte_level_drift(
    tmp_path, monkeypatch
):
    fit_dir = _write_dummy_final_fit(tmp_path, monkeypatch)
    path = fit_dir / "full_model.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        rfh.load_final_models()


def test_load_final_models_rejects_wrong_protocol_lineage(
    tmp_path, monkeypatch
):
    _write_dummy_final_fit(
        tmp_path, monkeypatch, bad_protocol=True
    )
    with pytest.raises(ValueError, match="manifest lineage"):
        rfh.load_final_models()


def test_holdout_boundary_is_berlin_midnight():
    assert pd.Timestamp(
        local_delivery_date_to_utc("2026-01-01")
    ) == pd.Timestamp("2025-12-31T23:00:00Z")
    assert pd.Timestamp(
        local_delivery_date_to_utc("2026-08-01")
    ) == pd.Timestamp("2026-07-31T22:00:00Z")


def _minimal_holdout_df(start_local, end_local_exclusive):
    start = local_delivery_date_to_utc(start_local)
    end = local_delivery_date_to_utc(end_local_exclusive)
    ts = pd.date_range(start, end, freq="1h", inclusive="left")
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "price_eur_mwh": np.ones(len(ts)),
        }
    )


def test_exact_coverage_passes_small_window(monkeypatch):
    monkeypatch.setattr(rfh, "HOLDOUT_START_LOCAL", "2026-01-01")
    monkeypatch.setattr(
        rfh, "HOLDOUT_END_LOCAL_EXCLUSIVE", "2026-01-06"
    )
    df = _minimal_holdout_df("2026-01-01", "2026-01-06")
    report = rfh.assert_exact_holdout_coverage(df)
    assert report["observed_hourly_rows"] == 120
    assert report["raw_delivery_days"] == 5


def test_exact_coverage_rejects_missing_hour(monkeypatch):
    monkeypatch.setattr(rfh, "HOLDOUT_START_LOCAL", "2026-01-01")
    monkeypatch.setattr(
        rfh, "HOLDOUT_END_LOCAL_EXCLUSIVE", "2026-01-03"
    )
    df = _minimal_holdout_df("2026-01-01", "2026-01-03")
    df = df.drop(index=[5]).reset_index(drop=True)
    with pytest.raises(ValueError, match="does not exactly match"):
        rfh.assert_exact_holdout_coverage(df)


def _trained_model_info():
    from xgboost import XGBRegressor

    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 2))
    y = 50 + 7 * X[:, 0] + rng.normal(0, 0.5, 300)
    model = XGBRegressor(n_estimators=30, max_depth=3)
    model.fit(X, y)
    return {"model": model, "predictor_cols": ["x1", "x2"]}


def _history_until(day_exclusive, n_days=90, scale=1.0):
    end_local = pd.Timestamp(day_exclusive).date()
    start_local = end_local - pd.Timedelta(days=n_days)
    rng = np.random.default_rng(5)
    rows = []
    for day in pd.date_range(start_local, periods=n_days, freq="D"):
        start = local_delivery_date_to_utc(day.strftime("%Y-%m-%d"))
        next_start = local_delivery_date_to_utc(
            (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )
        for ts in pd.date_range(
            start, next_start, freq="1h", inclusive="left"
        ):
            rows.append(
                {
                    "timestamp_utc": ts,
                    "prediction": 50.0,
                    "residual": float(rng.normal(0, scale)),
                }
            )
    return pd.DataFrame(rows)


def _day_df(local_day, actual_shift=0.0):
    day = pd.Timestamp(local_day)
    start = local_delivery_date_to_utc(day.strftime("%Y-%m-%d"))
    end = local_delivery_date_to_utc(
        (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    )
    ts = pd.date_range(start, end, freq="1h", inclusive="left")
    x1 = np.linspace(-2, 2, len(ts))
    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "x1": x1,
            "x2": np.zeros(len(ts)),
            "price_eur_mwh": 50 + actual_shift,
            "lag_24_pred": np.full(len(ts), 50.0),
            "lag_168_pred": np.full(len(ts), 49.0),
        }
    )


def test_hostile_actual_mutation_cannot_change_d_offsets():
    model_info = _trained_model_info()
    full_history = _history_until("2025-06-01")
    tier1_history = _history_until("2025-06-01")
    day = pd.Timestamp("2025-06-01").date()

    original, _, _ = rfh.build_day_forecast_frame(
        day,
        _day_df("2025-06-01", 0.0),
        model_info,
        model_info,
        full_history,
        tier1_history,
    )
    corrupted, _, _ = rfh.build_day_forecast_frame(
        day,
        _day_df("2025-06-01", 99999.0),
        model_info,
        model_info,
        full_history,
        tier1_history,
    )

    for col in (
        "xgboost_full_pred",
        "xgboost_tier1_pred",
        "full_q10_offset",
        "full_q50_offset",
        "full_q90_offset",
        "tier1_q10_offset",
        "tier1_q50_offset",
        "tier1_q90_offset",
    ):
        np.testing.assert_allclose(
            original[col].to_numpy(),
            corrupted[col].to_numpy(),
            equal_nan=True,
        )


def test_day_builder_does_not_mutate_histories():
    model_info = _trained_model_info()
    full_history = _history_until("2025-06-01")
    tier1_history = _history_until("2025-06-01")
    n_full = len(full_history)
    n_t1 = len(tier1_history)

    rfh.build_day_forecast_frame(
        pd.Timestamp("2025-06-01").date(),
        _day_df("2025-06-01"),
        model_info,
        model_info,
        full_history,
        tier1_history,
    )
    assert len(full_history) == n_full
    assert len(tier1_history) == n_t1


def test_positive_control_prior_day_changes_later_offset():
    model_info = _trained_model_info()
    tight = _history_until("2025-06-01", n_days=90, scale=0.1)
    later_day = pd.Timestamp("2025-06-02").date()

    base, _, _ = rfh.build_day_forecast_frame(
        later_day,
        _day_df("2025-06-02"),
        model_info,
        model_info,
        tight,
        tight,
    )

    start = local_delivery_date_to_utc("2025-06-01")
    end = local_delivery_date_to_utc("2025-06-02")
    prior_ts = pd.date_range(
        start, end, freq="1h", inclusive="left"
    )
    prior_residuals = pd.DataFrame(
        {
            "timestamp_utc": prior_ts,
            "prediction": 50.0,
            "residual": np.full(len(prior_ts), 999.0),
        }
    )
    extended = pd.concat(
        [tight, prior_residuals], ignore_index=True
    )

    changed, _, _ = rfh.build_day_forecast_frame(
        later_day,
        _day_df("2025-06-02"),
        model_info,
        model_info,
        extended,
        extended,
    )
    assert changed["full_q90_offset"].iloc[0] != pytest.approx(
        base["full_q90_offset"].iloc[0]
    )


def test_sequential_loop_processes_every_raw_day(monkeypatch):
    calls = []

    def fake_day(
        day_local_date,
        day_hourly_df,
        full_model_info,
        tier1_model_info,
        full_history,
        tier1_history,
    ):
        calls.append((day_local_date, len(full_history)))
        ts = pd.to_datetime(day_hourly_df["timestamp_utc"], utc=True)
        out = pd.DataFrame(
            {
                "timestamp_utc": ts,
                "delivery_date": day_local_date,
                "price_eur_mwh": day_hourly_df["price_eur_mwh"],
                "lag_24_pred": day_hourly_df["lag_24_pred"],
                "lag_168_pred": day_hourly_df["lag_168_pred"],
                "xgboost_full_pred": 1.0,
                "xgboost_tier1_pred": 1.0,
                "full_q10_offset": -1.0,
                "full_q50_offset": 0.0,
                "full_q90_offset": 1.0,
                "tier1_q10_offset": -1.0,
                "tier1_q50_offset": 0.0,
                "tier1_q90_offset": 1.0,
                "full_L": 0.0,
                "full_M": 1.0,
                "full_U": 2.0,
                "tier1_L": 0.0,
                "tier1_M": 1.0,
                "tier1_U": 2.0,
            }
        )
        residual = pd.DataFrame(
            {
                "timestamp_utc": ts,
                "prediction": 1.0,
                "residual": 1.0,
            }
        )
        return out, residual, residual.copy()

    monkeypatch.setattr(rfh, "build_day_forecast_frame", fake_day)

    start = local_delivery_date_to_utc("2026-01-01")
    end = local_delivery_date_to_utc("2026-01-03")
    ts = pd.date_range(start, end, freq="1h", inclusive="left")
    df = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "price_eur_mwh": 2.0,
            "price_lag_24h": 1.0,
            "price_lag_168h": 1.0,
        }
    )
    empty = pd.DataFrame(
        columns=["timestamp_utc", "prediction", "residual"]
    )
    result = rfh.build_sequential_holdout_forecasts(
        df, {}, {}, empty, empty.copy()
    )

    assert len(calls) == 2
    assert calls[0][1] == 0
    assert calls[1][1] == 24
    assert len(result) == 48


def test_common_mask_excludes_entire_day_for_one_missing_hour():
    start = local_delivery_date_to_utc("2026-01-01")
    end = local_delivery_date_to_utc("2026-01-03")
    ts = pd.date_range(start, end, freq="1h", inclusive="left")
    df = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "price_eur_mwh": 50.0,
            "lag_24_pred": 50.0,
            "lag_168_pred": 49.0,
            "xgboost_full_pred": 50.0,
            "xgboost_tier1_pred": 50.0,
            "full_L": 45.0,
            "full_U": 55.0,
            "tier1_L": 40.0,
            "tier1_U": 60.0,
        }
    )
    df.loc[5, "xgboost_full_pred"] = np.nan

    common = rfh.build_common_evaluation_days(df)
    assert len(common["common_days"]) == 1
    assert (
        common["coverage"][
            "days_excluded_for_incomplete_intraday_vector"
        ]
        == 1
    )


def test_baseline_predictions_create_lag24_and_lag168():
    df = pd.DataFrame(
        {
            "price_lag_24h": [1.0],
            "price_lag_168h": [2.0],
        }
    )
    out = rfh.baseline_predictions(df)
    assert out["lag_24_pred"].iloc[0] == 1.0
    assert out["lag_168_pred"].iloc[0] == 2.0


def test_primary_economics_calls_structural_gate(monkeypatch):
    called = {"value": False}

    def fake_verify(results):
        called["value"] = True

    monkeypatch.setattr(
        rfh, "verify_structural_invariants", fake_verify
    )

    start = local_delivery_date_to_utc("2026-01-01")
    end = local_delivery_date_to_utc("2026-01-02")
    ts = pd.date_range(start, end, freq="1h", inclusive="left")
    day = pd.Timestamp("2026-01-01").date()
    line = np.linspace(40, 80, len(ts))
    hourly = pd.DataFrame(
        {
            "timestamp_utc": ts,
            "delivery_date": day,
            "price_eur_mwh": line,
            "lag_24_pred": line,
            "lag_168_pred": line,
            "xgboost_full_pred": line,
            "xgboost_tier1_pred": line,
            "full_L": line - 1,
            "full_U": line + 1,
            "tier1_L": line - 1,
            "tier1_U": line + 1,
        }
    )
    _, result = rfh.run_primary_economics_on_common_days(
        hourly, [day]
    )
    assert called["value"] is True
    assert len(result) == 1


def test_full_end_to_end_synthetic_main(tmp_path, monkeypatch):
    """Full orchestration dry run on a tiny synthetic holdout window.

    This deliberately avoids real 2026 files and real frozen-model bytes,
    while exercising the same main() ordering, sequential update, common
    mask, economics, invariant gate, and save contract.
    """
    from xgboost import XGBRegressor
    from src.models import (
        ELASTICNET_PREDICTOR_COLS,
        ELASTICNET_TIER1_PREDICTOR_COLS,
    )

    monkeypatch.setattr(rfh, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rfh, "HOLDOUT_START_LOCAL", "2026-01-01")
    monkeypatch.setattr(
        rfh, "HOLDOUT_END_LOCAL_EXCLUSIVE", "2026-01-06"
    )

    rng = np.random.default_rng(21)

    full_model = XGBRegressor(n_estimators=5, max_depth=2)
    full_model.fit(
        rng.normal(size=(200, len(ELASTICNET_PREDICTOR_COLS))),
        rng.normal(50, 10, 200),
    )
    tier1_model = XGBRegressor(n_estimators=5, max_depth=2)
    tier1_model.fit(
        rng.normal(
            size=(200, len(ELASTICNET_TIER1_PREDICTOR_COLS))
        ),
        rng.normal(50, 10, 200),
    )

    monkeypatch.setattr(
        rfh,
        "load_final_models",
        lambda: {
            "models": {
                "full": {
                    "model": full_model,
                    "predictor_cols": ELASTICNET_PREDICTOR_COLS,
                },
                "tier1": {
                    "model": tier1_model,
                    "predictor_cols": ELASTICNET_TIER1_PREDICTOR_COLS,
                },
            },
            "manifest": {},
        },
    )

    full_history = _history_until(
        "2026-01-01", n_days=90, scale=5.0
    )
    tier1_history = _history_until(
        "2026-01-01", n_days=90, scale=8.0
    )

    def fake_history(pred_col):
        if pred_col == rfh.FULL_PRED_COL:
            return full_history.copy()
        return tier1_history.copy()

    monkeypatch.setattr(
        rfh, "build_initial_residual_history", fake_history
    )

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)

    start = local_delivery_date_to_utc("2026-01-01")
    end = local_delivery_date_to_utc("2026-01-06")
    ts = pd.date_range(start, end, freq="1h", inclusive="left")

    rows = []
    for t in ts:
        row = {
            "timestamp_utc": t,
            "price_eur_mwh": float(rng.normal(50, 10)),
        }
        for col in ELASTICNET_PREDICTOR_COLS:
            row[col] = float(rng.normal())
        rows.append(row)
    feature_df = pd.DataFrame(rows)

    feature_path = processed / rfh.INPUT_FILENAME
    feature_df.to_parquet(feature_path)

    monkeypatch.setattr(
        rfh,
        "verify_development_lineage",
        lambda path: {
            "current_input_sha256": rfh.sha256_file(path)
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_final_holdout.py", "synthetic_holdout_v1"],
    )

    rfh.main()

    out_dir = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfh.INPUT_STEM
        / "synthetic_holdout_v1"
    )
    assert (out_dir / "hourly_holdout_inputs.csv").exists()
    assert (out_dir / "per_day_results.csv").exists()
    assert (out_dir / "common_day_coverage.json").exists()
    assert (out_dir / "holdout_manifest.json").exists()

    manifest = json.loads(
        (out_dir / "holdout_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["holdout_used"] is True
    assert manifest["structural_invariants"] == "PASSED"
    assert manifest["exact_raw_holdout_coverage"] == "PASSED"
    assert manifest["raw_holdout_delivery_days"] == 5
    assert manifest["n_common_holdout_days"] == 5
    assert manifest["classification_complete"] is False

    hourly = pd.read_csv(out_dir / "hourly_holdout_inputs.csv")
    assert len(hourly) == 120
    assert "lag_24_pred" in hourly.columns
    assert "lag_168_pred" in hourly.columns
    assert "xgboost_full_pred" in hourly.columns
    assert "xgboost_tier1_pred" in hourly.columns
    assert "full_L" in hourly.columns
    assert "full_U" in hourly.columns
    assert "tier1_L" in hourly.columns
    assert "tier1_U" in hourly.columns
    assert "common_evaluation_day" in hourly.columns
    assert hourly["common_evaluation_day"].all()

    per_day = pd.read_csv(out_dir / "per_day_results.csv")
    assert len(per_day) == 5
    assert set(per_day["delivery_date"].astype(str)) == {
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
    }
