"""Tests for run_final_holdout_report.py.

No real 2026 artifacts are touched. Synthetic artifacts exercise:
  - frozen source/version lineage;
  - sensitivity-classification completeness;
  - hourly common-mask vs per-day consistency;
  - identical-row point metrics;
  - primary sensitivity/base-case total reproduction;
  - read-only decomposition/report generation;
  - overwrite protection.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_final_holdout_report as rfhr
from run_strategy_backtest import STRATEGIES, run_backtest_for_day
from src.clean import local_delivery_date_to_utc


def test_resolve_run_args_accepts_only_output_version():
    assert rfhr.resolve_run_args(["holdout_report_v1"]) == "holdout_report_v1"
    with pytest.raises(SystemExit):
        rfhr.resolve_run_args([])
    with pytest.raises(SystemExit):
        rfhr.resolve_run_args(["holdout_v1", "holdout_report_v1"])


def test_frozen_source_constants():
    assert rfhr.HOLDOUT_RUN_VERSION == "holdout_v1"
    assert rfhr.SENSITIVITY_RUN_VERSION == "holdout_sensitivity_v1"
    assert rfhr.PRIMARY_ETA_RT == 0.85
    assert rfhr.PRIMARY_C == 10.0
    assert rfhr.HOLDOUT_START_LOCAL == "2026-01-01"
    assert rfhr.HOLDOUT_END_LOCAL_EXCLUSIVE == "2026-08-01"


def _daily_hourly(day: str, *, price_shift: float = 0.0) -> pd.DataFrame:
    start = local_delivery_date_to_utc(day)
    next_day = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = local_delivery_date_to_utc(next_day)
    ts = pd.date_range(start, end, freq="1h", inclusive="left")

    n = len(ts)
    phase = np.linspace(-np.pi / 2, 3 * np.pi / 2, n)
    actual = 50 + price_shift + 35 * np.sin(phase)

    # Deliberately different point forecasts so metrics are non-degenerate.
    lag24 = actual + 6.0
    lag168 = actual - 9.0
    full = actual + np.linspace(-3.0, 3.0, n)
    tier1 = actual + np.linspace(-5.0, 5.0, n)

    return pd.DataFrame(
        {
            "timestamp_utc": ts,
            "price_eur_mwh": actual,
            "lag_24_pred": lag24,
            "lag_168_pred": lag168,
            "xgboost_full_pred": full,
            "xgboost_tier1_pred": tier1,
            "full_L": full - 4.0,
            "full_U": full + 4.0,
            "tier1_L": tier1 - 7.0,
            "tier1_U": tier1 + 7.0,
            "common_evaluation_day": True,
        }
    )


def _make_per_day(hourly: pd.DataFrame) -> pd.DataFrame:
    local_dates = (
        pd.to_datetime(hourly["timestamp_utc"], utc=True)
        .dt.tz_convert("Europe/Berlin")
        .dt.date
    )
    work = hourly.copy()
    work["delivery_date"] = local_dates

    rows = []
    for day, group in work.groupby("delivery_date", sort=True):
        result = run_backtest_for_day(group, 0.85, 10.0)
        row = {
            "delivery_date": str(day),
            "n_hours": len(group),
            "oracle_pnl": result["oracle"],
        }
        for s in STRATEGIES:
            row[f"{s}_traded"] = result[s]["traded"]
            row[f"{s}_i"] = result[s]["i"]
            row[f"{s}_j"] = result[s]["j"]
            row[f"{s}_net_pnl"] = result[s]["net_pnl"]
            row[f"{s}_gross_pnl"] = result[s]["gross_pnl"]
        rows.append(row)
    return pd.DataFrame(rows)


def _grid_from_per_day(per_day: pd.DataFrame, verdict="CONFIRMATION"):
    totals = {s: float(per_day[f"{s}_net_pnl"].sum()) for s in STRATEGIES}
    rows = []
    for eta in [0.70, 0.85, 0.92]:
        for c in [5, 10, 15]:
            # Non-primary cells need only be structurally complete for report
            # tests. Keep all delta_forecast positive for CONFIRMATION.
            offset = (eta - 0.70) * 100 + (15 - c)
            row = {
                "eta_rt": eta,
                "c": c,
                **{f"{s}_total_net_pnl": totals[s] for s in STRATEGIES},
                "delta_forecast": 10.0 + offset,
                "delta_uncertainty": -2.0,
                "delta_tier1": -3.0,
                "delta_tier1_u": -4.0,
            }
            rows.append(row)

    grid = pd.DataFrame(rows)

    # Primary cell MUST exactly reflect saved per-day totals/deltas.
    mask = np.isclose(grid["eta_rt"], 0.85) & np.isclose(grid["c"], 10)
    for s in STRATEGIES:
        grid.loc[mask, f"{s}_total_net_pnl"] = totals[s]
    grid.loc[mask, "delta_forecast"] = totals["S2"] - totals["S1"]
    grid.loc[mask, "delta_uncertainty"] = totals["S3"] - totals["S2"]
    grid.loc[mask, "delta_tier1"] = totals["S4"] - totals["S2"]
    grid.loc[mask, "delta_tier1_u"] = totals["S5"] - totals["S3"]

    n_positive = int((grid["delta_forecast"] > 0).sum())
    if verdict == "CONFIRMATION" and n_positive != 9:
        # For synthetic fixtures, force non-primary cells sufficiently high;
        # the primary cell remains genuine.
        nonprimary = ~mask
        grid.loc[nonprimary, "delta_forecast"] = 100.0
        n_positive = int((grid["delta_forecast"] > 0).sum())

    return grid, n_positive


def _write_frozen_artifacts(
    tmp_path: Path,
    *,
    bad_protocol=False,
    sensitivity_complete=True,
    verdict="CONFIRMATION",
):
    holdout_dir = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfhr.INPUT_STEM
        / rfhr.HOLDOUT_RUN_VERSION
    )
    sensitivity_dir = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfhr.INPUT_STEM
        / rfhr.SENSITIVITY_RUN_VERSION
    )
    holdout_dir.mkdir(parents=True)
    sensitivity_dir.mkdir(parents=True)

    hourly = pd.concat(
        [
            _daily_hourly("2026-01-01", price_shift=0.0),
            _daily_hourly("2026-01-02", price_shift=3.0),
        ],
        ignore_index=True,
    )
    hourly.to_csv(holdout_dir / "hourly_holdout_inputs.csv", index=False)

    per_day = _make_per_day(hourly)
    per_day.to_csv(holdout_dir / "per_day_results.csv", index=False)

    holdout_manifest = {
        "output_run_version": rfhr.HOLDOUT_RUN_VERSION,
        "holdout_used": True,
        "classification_complete": False,
        "holdout_start_local": rfhr.HOLDOUT_START_LOCAL,
        "holdout_end_local_exclusive": rfhr.HOLDOUT_END_LOCAL_EXCLUSIVE,
        "eta_rt_primary": rfhr.PRIMARY_ETA_RT,
        "c_primary": rfhr.PRIMARY_C,
        "strategies": list(STRATEGIES),
        "structural_invariants": "PASSED",
        "exact_raw_holdout_coverage": "PASSED",
        "protocol_tag": rfhr.PROTOCOL_TAG,
        "protocol_commit": "wrong" if bad_protocol else rfhr.PROTOCOL_COMMIT,
        "development_evidence_tag": rfhr.DEVELOPMENT_TAG,
        "development_evidence_commit": rfhr.DEVELOPMENT_COMMIT,
        "final_model_fit_tag": rfhr.FINAL_MODEL_FIT_TAG,
        "final_model_fit_commit": rfhr.FINAL_MODEL_FIT_COMMIT,
        "raw_holdout_delivery_days": 212,
        "n_common_holdout_days": len(per_day),
    }
    (holdout_dir / "holdout_manifest.json").write_text(
        json.dumps(holdout_manifest), encoding="utf-8"
    )

    grid, n_positive = _grid_from_per_day(per_day, verdict=verdict)
    grid.to_csv(sensitivity_dir / "sensitivity_grid.csv", index=False)

    # The manifest must be mechanically consistent with the persisted grid.
    primary_row = grid[
        np.isclose(grid["eta_rt"], 0.85)
        & np.isclose(grid["c"], 10)
    ].iloc[0]
    mechanical_verdict, mechanical_n_positive, mechanical_primary = (
        rfhr.classify_section_8(grid)
    )

    sensitivity_manifest = {
        "source_holdout_run_version": rfhr.HOLDOUT_RUN_VERSION,
        "base_case_reproduction": "PASSED",
        "classification_complete": sensitivity_complete,
        "holdout_used": True,
        "n_grid_cells": 9,
        "source_holdout_protocol_commit": rfhr.PROTOCOL_COMMIT,
        "source_final_model_fit_commit": rfhr.FINAL_MODEL_FIT_COMMIT,
        "section_8_verdict": mechanical_verdict,
        "n_positive_delta_forecast": mechanical_n_positive,
        "primary_delta_forecast": mechanical_primary,
    }
    (sensitivity_dir / "sensitivity_manifest.json").write_text(
        json.dumps(sensitivity_manifest), encoding="utf-8"
    )

    return hourly, per_day, grid


def test_load_frozen_artifacts_accepts_consistent_sources(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rfhr, "REPO_ROOT", tmp_path)
    hourly, per_day, grid = _write_frozen_artifacts(tmp_path)

    loaded = rfhr.load_frozen_artifacts()
    assert len(loaded["hourly"]) == len(hourly)
    assert len(loaded["per_day"]) == len(per_day)
    assert len(loaded["grid"]) == len(grid)


def test_loader_rejects_wrong_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(rfhr, "REPO_ROOT", tmp_path)
    _write_frozen_artifacts(tmp_path, bad_protocol=True)

    with pytest.raises(
        ValueError, match="FINAL HOLDOUT REPORT LINEAGE CHECK FAILED"
    ):
        rfhr.load_frozen_artifacts()


def test_loader_rejects_incomplete_sensitivity(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rfhr, "REPO_ROOT", tmp_path)
    _write_frozen_artifacts(
        tmp_path, sensitivity_complete=False
    )

    with pytest.raises(
        ValueError, match="FINAL HOLDOUT REPORT LINEAGE CHECK FAILED"
    ):
        rfhr.load_frozen_artifacts()


def test_loader_rejects_manifest_verdict_that_disagrees_with_grid(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rfhr, "REPO_ROOT", tmp_path)
    _write_frozen_artifacts(tmp_path)

    path = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfhr.INPUT_STEM
        / rfhr.SENSITIVITY_RUN_VERSION
        / "sensitivity_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    actual = manifest["section_8_verdict"]
    manifest["section_8_verdict"] = (
        "FAILURE" if actual != "FAILURE" else "CONFIRMATION"
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError, match="verdict does not reproduce"
    ):
        rfhr.load_frozen_artifacts()


def test_loader_rejects_hourly_common_day_mismatch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rfhr, "REPO_ROOT", tmp_path)
    _write_frozen_artifacts(tmp_path)

    path = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfhr.INPUT_STEM
        / rfhr.HOLDOUT_RUN_VERSION
        / "hourly_holdout_inputs.csv"
    )
    hourly = pd.read_csv(path)
    # Exclude one whole saved day from the hourly common mask.
    hourly.loc[:23, "common_evaluation_day"] = False
    hourly.to_csv(path, index=False)

    with pytest.raises(
        ValueError, match="Hourly common-day mask differs"
    ):
        rfhr.load_frozen_artifacts()


def test_loader_rejects_primary_grid_total_mismatch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rfhr, "REPO_ROOT", tmp_path)
    _write_frozen_artifacts(tmp_path)

    path = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfhr.INPUT_STEM
        / rfhr.SENSITIVITY_RUN_VERSION
        / "sensitivity_grid.csv"
    )
    grid = pd.read_csv(path)
    mask = np.isclose(grid["eta_rt"], 0.85) & np.isclose(grid["c"], 10)
    grid.loc[mask, "S2_total_net_pnl"] += 999.0
    grid.to_csv(path, index=False)

    with pytest.raises(
        ValueError, match="does not reproduce holdout per-day total"
    ):
        rfhr.load_frozen_artifacts()


def test_point_metrics_use_one_identical_finite_row_set():
    hourly = pd.concat(
        [
            _daily_hourly("2026-01-01"),
            _daily_hourly("2026-01-02"),
        ],
        ignore_index=True,
    )
    # One row becomes unavailable for one model, therefore must be removed
    # for ALL models from the point-comparison population.
    hourly.loc[0, "xgboost_tier1_pred"] = np.nan

    metrics, common_rows = rfhr.compute_point_forecast_metrics(hourly)
    assert len(common_rows) == len(hourly) - 1
    assert metrics["n"].nunique() == 1
    assert metrics["n"].iloc[0] == len(hourly) - 1
    assert set(metrics["model"]) == {
        "lag24",
        "lag168",
        "xgboost_full",
        "xgboost_tier1",
    }
    assert {"mae", "rmse", "median_ae", "n"}.issubset(metrics.columns)


def test_attach_market_flags_from_saved_hourly_actuals():
    hourly = _daily_hourly("2026-01-01")
    hourly.loc[0, "price_eur_mwh"] = -5.0
    hourly.loc[1, "price_eur_mwh"] = 250.0
    hourly.loc[2, "price_eur_mwh"] = 550.0
    per_day = _make_per_day(hourly)

    out = rfhr.attach_market_flags(per_day, hourly)
    row = out.iloc[0]
    assert bool(row["is_negative_price_day"]) is True
    assert bool(row["is_gt200_day"]) is True
    assert bool(row["is_gt500_day"]) is True


def test_full_end_to_end_report_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rfhr, "REPO_ROOT", tmp_path)
    _write_frozen_artifacts(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_final_holdout_report.py", "holdout_report_v1"],
    )
    rfhr.main()

    out_dir = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfhr.INPUT_STEM
        / "holdout_report_v1"
    )
    expected = {
        "point_forecast_metrics.csv",
        "primary_strategy_table.csv",
        "sensitivity_grid.csv",
        "decomposition_results.json",
        "holdout_diagnostics.json",
        "holdout_report.txt",
        "holdout_report_manifest.json",
    }
    assert expected.issubset({p.name for p in out_dir.iterdir()})

    manifest = json.loads(
        (out_dir / "holdout_report_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["holdout_used"] is True
    assert manifest["decision_recomputation_performed"] is False
    assert manifest["forecast_recomputation_performed"] is False
    assert manifest["section_8_verdict"] in {
        "CONFIRMATION",
        "PARTIAL CONFIRMATION",
        "FAILURE",
    }

    metrics = pd.read_csv(out_dir / "point_forecast_metrics.csv")
    assert len(metrics) == 4
    assert metrics["n"].nunique() == 1


def test_report_output_overwrite_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rfhr, "REPO_ROOT", tmp_path)
    _write_frozen_artifacts(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_final_holdout_report.py", "holdout_report_v1"],
    )
    rfhr.main()

    with pytest.raises(FileExistsError):
        rfhr.main()
