"""Integrated Section 9 pre-holdout rehearsal.

Run ONLY as a test:

    python -m pytest tests/test_pre_holdout_integrated_dryrun.py -v -s

This test never reads the real processed 2026 parquet and never writes to the
real repository outputs. It creates one shared synthetic surrogate holdout in
pytest's tmp_path and passes the actual persisted artifacts through the frozen
chain:

    run_final_holdout
      -> run_final_holdout_sensitivity
      -> run_final_holdout_report
      -> run_final_holdout_tailrisk

Only production-data/model-loading boundaries are substituted. The scientific
kernels and downstream provenance/reproduction checks remain real.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_final_holdout as rfh
import run_final_holdout_sensitivity as rfhs
import run_final_holdout_report as rfhr
import run_final_holdout_tailrisk as rfht

from src.clean import local_delivery_date_to_utc
from src.models import (
    ELASTICNET_PREDICTOR_COLS,
    ELASTICNET_TIER1_PREDICTOR_COLS,
)
from run_strategy_backtest import STRATEGIES


SURROGATE_START = "2026-01-01"
SURROGATE_END_EXCLUSIVE = "2026-01-31"
EXCLUDED_DAY = "2026-01-12"


def _make_residual_history(
    day_exclusive: str,
    *,
    n_days: int,
    scale: float,
    seed: int,
) -> pd.DataFrame:
    end_local = pd.Timestamp(day_exclusive).date()
    start_local = end_local - pd.Timedelta(days=n_days)
    rng = np.random.default_rng(seed)

    rows = []
    for day in pd.date_range(start_local, periods=n_days, freq="D"):
        start = local_delivery_date_to_utc(day.strftime("%Y-%m-%d"))
        end = local_delivery_date_to_utc(
            (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )
        for ts in pd.date_range(start, end, freq="1h", inclusive="left"):
            rows.append(
                {
                    "timestamp_utc": ts,
                    "prediction": 50.0,
                    "residual": float(rng.normal(0.0, scale)),
                }
            )
    return pd.DataFrame(rows)


def _make_surrogate_features() -> pd.DataFrame:
    rng = np.random.default_rng(20260913)

    start = local_delivery_date_to_utc(SURROGATE_START)
    end = local_delivery_date_to_utc(SURROGATE_END_EXCLUSIVE)
    ts = pd.date_range(start, end, freq="1h", inclusive="left")

    local = ts.tz_convert("Europe/Berlin")
    hours = local.hour.to_numpy()
    day_index = np.array(
        [(d - local.date[0]).days for d in local.date], dtype=int
    )

    price = (
        55.0
        + 22.0 * np.sin(2.0 * np.pi * hours / 24.0 - np.pi / 2.0)
        + 8.0 * np.sin(2.0 * np.pi * day_index / 9.0)
        + rng.normal(0.0, 3.0, len(ts))
    )
    price[(hours == 19) & (day_index % 7 == 0)] += 90.0
    price[(hours == 3) & (day_index % 8 == 0)] -= 70.0

    rows = []
    for i, t in enumerate(ts):
        row = {
            "timestamp_utc": t,
            "price_eur_mwh": float(price[i]),
        }
        for j, col in enumerate(ELASTICNET_PREDICTOR_COLS):
            row[col] = (
                20.0
                + 0.5 * j
                + 0.2 * hours[i]
                + 0.1 * day_index[i]
                + float(rng.normal(0.0, 0.5))
            )

        row["price_lag_24h"] = float(price[max(0, i - 24)])
        row["price_lag_168h"] = float(price[max(0, i - 168)])
        rows.append(row)

    df = pd.DataFrame(rows)

    local_dates = (
        pd.to_datetime(df["timestamp_utc"], utc=True)
        .dt.tz_convert("Europe/Berlin")
        .dt.date.astype(str)
    )
    excluded_indices = df.index[local_dates == EXCLUDED_DAY]
    assert len(excluded_indices) == 24
    df.loc[excluded_indices[7], "renewables_forecast_mw"] = np.nan

    return df


def _make_surrogate_models():
    from xgboost import XGBRegressor

    rng = np.random.default_rng(42)

    x_full = rng.normal(size=(500, len(ELASTICNET_PREDICTOR_COLS)))
    y_full = (
        50.0
        + 4.0 * x_full[:, 0]
        - 2.0 * x_full[:, 4]
        + rng.normal(0.0, 2.0, 500)
    )
    full = XGBRegressor(
        n_estimators=12,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.8,
        objective="reg:absoluteerror",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    full.fit(x_full, y_full)

    x_t1 = rng.normal(size=(500, len(ELASTICNET_TIER1_PREDICTOR_COLS)))
    y_t1 = (
        50.0
        + 3.5 * x_t1[:, 0]
        - 1.5 * x_t1[:, 4]
        + rng.normal(0.0, 2.5, 500)
    )
    tier1 = XGBRegressor(
        n_estimators=12,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.8,
        objective="reg:absoluteerror",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    tier1.fit(x_t1, y_t1)

    return {
        "models": {
            "full": {
                "model": full,
                "predictor_cols": ELASTICNET_PREDICTOR_COLS,
            },
            "tier1": {
                "model": tier1,
                "predictor_cols": ELASTICNET_TIER1_PREDICTOR_COLS,
            },
        },
        "manifest": {"synthetic_section9_rehearsal": True},
    }


def _patch_shared_repo(monkeypatch, tmp_path: Path) -> None:
    for module in (rfh, rfhs, rfhr, rfht):
        monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    monkeypatch.setattr(rfh, "HOLDOUT_START_LOCAL", SURROGATE_START)
    monkeypatch.setattr(
        rfh, "HOLDOUT_END_LOCAL_EXCLUSIVE", SURROGATE_END_EXCLUSIVE
    )
    monkeypatch.setattr(rfhr, "HOLDOUT_START_LOCAL", SURROGATE_START)
    monkeypatch.setattr(
        rfhr, "HOLDOUT_END_LOCAL_EXCLUSIVE", SURROGATE_END_EXCLUSIVE
    )
    monkeypatch.setattr(rfht, "HOLDOUT_START_LOCAL", SURROGATE_START)
    monkeypatch.setattr(
        rfht, "HOLDOUT_END_LOCAL_EXCLUSIVE", SURROGATE_END_EXCLUSIVE
    )


def test_section9_integrated_surrogate_chain(tmp_path, monkeypatch):
    _patch_shared_repo(monkeypatch, tmp_path)

    # 0. ONE shared surrogate input and pre-holdout state.
    cfg = rfh.load_config()
    processed_dir = tmp_path / cfg["data"]["processed_dir"]
    processed_dir.mkdir(parents=True)

    feature_df = _make_surrogate_features()
    feature_path = processed_dir / rfh.INPUT_FILENAME
    feature_df.to_parquet(feature_path, index=False)

    surrogate_models = _make_surrogate_models()
    monkeypatch.setattr(
        rfh, "load_final_models", lambda: surrogate_models
    )

    full_history = _make_residual_history(
        SURROGATE_START, n_days=90, scale=5.0, seed=100
    )
    tier1_history = _make_residual_history(
        SURROGATE_START, n_days=90, scale=8.0, seed=200
    )

    def fake_history(pred_col):
        if pred_col == rfh.FULL_PRED_COL:
            return full_history.copy()
        if pred_col == rfh.TIER1_PRED_COL:
            return tier1_history.copy()
        raise AssertionError(
            f"Unexpected residual prediction column {pred_col!r}"
        )

    monkeypatch.setattr(
        rfh, "build_initial_residual_history", fake_history
    )
    monkeypatch.setattr(
        rfh,
        "verify_development_lineage",
        lambda path: {
            "current_input_sha256": rfh.sha256_file(path),
            "synthetic_section9_rehearsal": True,
        },
    )

    # 1. Holdout runner on surrogate data.
    monkeypatch.setattr(
        sys, "argv", ["run_final_holdout.py", "holdout_v1"]
    )
    rfh.main()

    holdout_dir = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfh.INPUT_STEM
        / "holdout_v1"
    )
    assert (holdout_dir / "hourly_holdout_inputs.csv").exists()
    assert (holdout_dir / "per_day_results.csv").exists()
    assert (holdout_dir / "holdout_manifest.json").exists()

    holdout_manifest = json.loads(
        (holdout_dir / "holdout_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert holdout_manifest["structural_invariants"] == "PASSED"
    assert holdout_manifest["exact_raw_holdout_coverage"] == "PASSED"
    assert holdout_manifest["raw_holdout_delivery_days"] == 30
    assert holdout_manifest["n_common_holdout_days"] == 29
    assert holdout_manifest["classification_complete"] is False

    hourly = pd.read_csv(holdout_dir / "hourly_holdout_inputs.csv")
    hourly_local_date = (
        pd.to_datetime(hourly["timestamp_utc"], utc=True)
        .dt.tz_convert("Europe/Berlin")
        .dt.date.astype(str)
    )

    excluded = hourly[hourly_local_date == EXCLUDED_DAY]
    assert len(excluded) == 24
    excluded_common = (
        excluded["common_evaluation_day"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    assert int(excluded_common.sum()) == 0
    assert excluded["xgboost_full_pred"].isna().sum() == 1
    assert excluded["xgboost_tier1_pred"].notna().all()

    later = hourly[hourly_local_date == "2026-01-13"]
    assert later["full_q10_offset"].notna().all()
    assert later["full_q90_offset"].notna().all()
    assert later["tier1_q10_offset"].notna().all()
    assert later["tier1_q90_offset"].notna().all()

    per_day = pd.read_csv(holdout_dir / "per_day_results.csv")
    assert len(per_day) == 29
    assert EXCLUDED_DAY not in set(per_day["delivery_date"].astype(str))

    # 2. Sensitivity consumes the actual saved holdout artifacts.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_holdout_sensitivity.py",
            "holdout_sensitivity_v1",
        ],
    )
    rfhs.main()

    sensitivity_dir = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfhs.INPUT_STEM
        / "holdout_sensitivity_v1"
    )
    grid = pd.read_csv(sensitivity_dir / "sensitivity_grid.csv")
    sensitivity_manifest = json.loads(
        (sensitivity_dir / "sensitivity_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(grid) == 9
    assert not grid[["eta_rt", "c"]].duplicated().any()
    assert sensitivity_manifest["base_case_reproduction"] == "PASSED"
    assert sensitivity_manifest["classification_complete"] is True
    assert sensitivity_manifest["n_common_holdout_days"] == 29
    assert sensitivity_manifest["section_8_verdict"] in {
        "CONFIRMATION",
        "PARTIAL CONFIRMATION",
        "FAILURE",
    }

    # 3. Report consumes the same holdout + sensitivity artifacts.
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_final_holdout_report.py", "holdout_report_v1"],
    )
    rfhr.main()

    report_dir = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfhr.INPUT_STEM
        / "holdout_report_v1"
    )
    report_manifest = json.loads(
        (report_dir / "holdout_report_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    point_metrics = pd.read_csv(
        report_dir / "point_forecast_metrics.csv"
    )
    decomposition = json.loads(
        (report_dir / "decomposition_results.json").read_text(
            encoding="utf-8"
        )
    )

    assert report_manifest["source_holdout_run_version"] == "holdout_v1"
    assert (
        report_manifest["source_sensitivity_run_version"]
        == "holdout_sensitivity_v1"
    )
    assert report_manifest["section_8_verdict"] == sensitivity_manifest[
        "section_8_verdict"
    ]
    assert report_manifest["n_common_economic_days"] == 29
    assert report_manifest["decision_recomputation_performed"] is False
    assert report_manifest["forecast_recomputation_performed"] is False

    assert len(point_metrics) == 4
    assert point_metrics["n"].nunique() == 1
    assert int(point_metrics["n"].iloc[0]) == len(hourly) - 1
    assert set(point_metrics["model"]) == {
        "lag24",
        "lag168",
        "xgboost_full",
        "xgboost_tier1",
    }
    assert {"S2_vs_S3", "S4_vs_S5"}.issubset(decomposition)

    # 4. Tail-risk consumes the same primary per-day holdout P&L.
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_final_holdout_tailrisk.py", "holdout_tailrisk_v1"],
    )
    rfht.main()

    risk_dir = (
        tmp_path
        / "outputs"
        / "risk"
        / rfht.INPUT_STEM
        / "holdout_tailrisk_v1"
    )
    var_es = pd.read_csv(risk_dir / "var_es_results.csv")
    risk_manifest = json.loads(
        (risk_dir / "tailrisk_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(var_es) == len(STRATEGIES) * 2
    assert set(var_es["strategy"]) == set(STRATEGIES)
    assert set(np.round(var_es["confidence_level"], 2)) == {0.95, 0.99}
    assert set(var_es["n_total"].astype(int)) == {29}
    low_precision = (
        var_es["low_precision"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    assert bool(low_precision.all())

    assert risk_manifest["source_holdout_run_version"] == "holdout_v1"
    assert risk_manifest["n_common_holdout_days"] == 29
    assert risk_manifest["decision_recomputation_performed"] is False
    assert risk_manifest["forecast_recomputation_performed"] is False

    # 5. Cross-stage provenance remains one coherent chain.
    assert sensitivity_manifest["source_holdout_run_version"] == "holdout_v1"
    assert report_manifest["source_holdout_run_version"] == "holdout_v1"
    assert risk_manifest["source_holdout_run_version"] == "holdout_v1"

    assert (
        sensitivity_manifest["source_holdout_protocol_commit"]
        == rfh.PROTOCOL_COMMIT
    )
    assert report_manifest["protocol_commit"] == rfh.PROTOCOL_COMMIT
    assert risk_manifest["protocol_commit"] == rfh.PROTOCOL_COMMIT

    print("\n" + "=" * 78)
    print("SECTION 9 INTEGRATED SURROGATE REHEARSAL: PASSED")
    print("One shared synthetic artifact chain completed end-to-end.")
    print("Raw surrogate delivery days: 30")
    print("Common economic delivery days: 29")
    print(f"Deliberately excluded common-mask day: {EXCLUDED_DAY}")
    print(
        "Mechanical synthetic verdict (not scientific evidence): "
        f"{sensitivity_manifest['section_8_verdict']}"
    )
    print("No real 2026 project artifact was read or written.")
    print("=" * 78)
