"""Tests for run_final_holdout_sensitivity.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_final_holdout_sensitivity as rfhs
from src.clean import add_local_time_columns, local_delivery_date_to_utc


def test_resolve_run_args_accepts_only_output_version():
    assert (
        rfhs.resolve_run_args(["holdout_sensitivity_v1"])
        == "holdout_sensitivity_v1"
    )
    with pytest.raises(SystemExit):
        rfhs.resolve_run_args([])
    with pytest.raises(SystemExit):
        rfhs.resolve_run_args(["holdout_v1", "holdout_sensitivity_v1"])


def test_frozen_grid_contract():
    from run_strategy_sensitivity import ETA_RT_GRID, C_GRID

    assert rfhs.ETA_RT_GRID is ETA_RT_GRID
    assert rfhs.C_GRID is C_GRID
    assert rfhs.ETA_RT_GRID == [0.70, 0.85, 0.92]
    assert rfhs.C_GRID == [5, 10, 15]
    assert rfhs.HOLDOUT_RUN_VERSION == "holdout_v1"
    rfhs.assert_frozen_grid_contract()


def _synthetic_common_df(n_days=3):
    rows = []
    for d in range(n_days):
        local_day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)
        start = local_delivery_date_to_utc(local_day.strftime("%Y-%m-%d"))
        end = local_delivery_date_to_utc(
            (local_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )
        ts = pd.date_range(start, end, freq="1h", inclusive="left")
        n = len(ts)
        phase = np.linspace(-np.pi / 2, 3 * np.pi / 2, n)
        price = 50 + 30 * np.sin(phase)

        for t, p in zip(ts, price):
            rows.append(
                {
                    "timestamp_utc": t,
                    "price_eur_mwh": p,
                    "lag_24_pred": p,
                    "lag_168_pred": p,
                    "xgboost_full_pred": p,
                    "xgboost_tier1_pred": p,
                    "full_L": p - 2,
                    "full_U": p + 2,
                    "tier1_L": p - 4,
                    "tier1_U": p + 4,
                    "common_evaluation_day": True,
                }
            )
    return add_local_time_columns(pd.DataFrame(rows))


def test_compute_cell_returns_one_row_per_day():
    cell = rfhs.compute_cell(_synthetic_common_df(3), 0.85, 10.0)
    assert len(cell) == 3
    assert set(cell["delivery_date"]) == {
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
    }


def test_compute_cell_has_genuine_trade():
    cell = rfhs.compute_cell(_synthetic_common_df(1), 0.85, 10.0)
    assert bool(cell.iloc[0]["S2_traded"]) is True


def test_different_eta_c_changes_economics():
    df = _synthetic_common_df(1)
    low = rfhs.compute_cell(df, 0.70, 15.0)
    high = rfhs.compute_cell(df, 0.92, 5.0)
    assert low.iloc[0]["S2_net_pnl"] != pytest.approx(
        high.iloc[0]["S2_net_pnl"]
    )


def test_base_case_reproduction_passes_tiny_csv_noise():
    computed = rfhs.compute_cell(_synthetic_common_df(2), 0.85, 10.0)
    saved = computed.copy()
    saved["S2_net_pnl"] = saved["S2_net_pnl"] + 1e-11
    rfhs.verify_base_case_reproduction(computed, saved)


def test_base_case_reproduction_rejects_material_mismatch():
    computed = rfhs.compute_cell(_synthetic_common_df(2), 0.85, 10.0)
    saved = computed.copy()
    saved.loc[0, "S2_net_pnl"] += 1.0

    with pytest.raises(
        AssertionError, match="BASE-CASE REPRODUCTION FAILED"
    ):
        rfhs.verify_base_case_reproduction(computed, saved)


def test_base_case_reproduction_rejects_day_mismatch():
    computed = rfhs.compute_cell(_synthetic_common_df(2), 0.85, 10.0)
    saved = computed.copy()
    saved["delivery_date"] = ["2099-01-01", "2099-01-02"]

    with pytest.raises(
        AssertionError, match="BASE-CASE REPRODUCTION FAILED"
    ):
        rfhs.verify_base_case_reproduction(computed, saved)


def _write_holdout_artifacts(
    tmp_path,
    *,
    primary_eta=0.85,
    primary_c=10.0,
    holdout_used=True,
    protocol_commit=None,
):
    holdout_dir = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfhs.INPUT_STEM
        / rfhs.HOLDOUT_RUN_VERSION
    )
    holdout_dir.mkdir(parents=True)

    hourly = _synthetic_common_df(2).copy()

    # Add a third raw day explicitly excluded by the frozen common mask.
    extra = _synthetic_common_df(1).copy()
    extra["timestamp_utc"] = extra["timestamp_utc"] + pd.Timedelta(days=2)
    extra["common_evaluation_day"] = False
    extra = add_local_time_columns(extra)

    hourly = pd.concat([hourly, extra], ignore_index=True)
    hourly.drop(
        columns=["delivery_date", "local_time"], errors="ignore"
    ).to_csv(
        holdout_dir / "hourly_holdout_inputs.csv", index=False
    )

    common = add_local_time_columns(
        hourly[hourly["common_evaluation_day"]].copy()
    )
    per_day = rfhs.compute_cell(common, rfhs.PRIMARY_ETA_RT, rfhs.PRIMARY_C)
    per_day.to_csv(holdout_dir / "per_day_results.csv", index=False)

    manifest = {
        "output_run_version": rfhs.HOLDOUT_RUN_VERSION,
        "holdout_used": holdout_used,
        "classification_complete": False,
        "eta_rt_primary": primary_eta,
        "c_primary": primary_c,
        "strategies": rfhs.EXPECTED_STRATEGIES,
        "structural_invariants": "PASSED",
        "exact_raw_holdout_coverage": "PASSED",
        "common_mask_method": rfhs.EXPECTED_COMMON_MASK_METHOD,
        "protocol_tag": rfhs.PROTOCOL_TAG,
        "protocol_commit": (
            rfhs.PROTOCOL_COMMIT
            if protocol_commit is None
            else protocol_commit
        ),
        "development_evidence_tag": rfhs.DEVELOPMENT_TAG,
        "development_evidence_commit": rfhs.DEVELOPMENT_COMMIT,
        "final_model_fit_tag": rfhs.FINAL_MODEL_FIT_TAG,
        "final_model_fit_commit": rfhs.FINAL_MODEL_FIT_COMMIT,
        "n_common_holdout_days": 2,
    }
    (holdout_dir / "holdout_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return holdout_dir


def test_loader_uses_only_common_evaluation_days(tmp_path, monkeypatch):
    monkeypatch.setattr(rfhs, "REPO_ROOT", tmp_path)
    _write_holdout_artifacts(tmp_path)

    loaded = rfhs.load_holdout_artifacts()
    assert len(loaded["hourly_all"]) == 72
    assert len(loaded["common_hourly"]) == 48
    assert loaded["common_hourly"]["delivery_date"].nunique() == 2


def test_loader_rejects_non_primary_eta(tmp_path, monkeypatch):
    monkeypatch.setattr(rfhs, "REPO_ROOT", tmp_path)
    _write_holdout_artifacts(tmp_path, primary_eta=0.70)

    with pytest.raises(
        ValueError, match="HOLDOUT SOURCE LINEAGE CHECK FAILED"
    ):
        rfhs.load_holdout_artifacts()


def test_loader_rejects_holdout_used_false(tmp_path, monkeypatch):
    monkeypatch.setattr(rfhs, "REPO_ROOT", tmp_path)
    _write_holdout_artifacts(tmp_path, holdout_used=False)

    with pytest.raises(
        ValueError, match="HOLDOUT SOURCE LINEAGE CHECK FAILED"
    ):
        rfhs.load_holdout_artifacts()


def test_loader_rejects_wrong_protocol_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(rfhs, "REPO_ROOT", tmp_path)
    _write_holdout_artifacts(tmp_path, protocol_commit="wrong")

    with pytest.raises(
        ValueError, match="HOLDOUT SOURCE LINEAGE CHECK FAILED"
    ):
        rfhs.load_holdout_artifacts()


def test_structural_adapter_preserves_delivery_date():
    cell = rfhs.compute_cell(_synthetic_common_df(1), 0.85, 10.0)
    nested = rfhs.cell_to_structural_results(cell)
    assert nested[0]["delivery_date"] == "2026-01-01"
    rfhs.verify_structural_invariants(nested)


def test_classification_mechanical_rules():
    rows = []
    for eta in rfhs.ETA_RT_GRID:
        for c in rfhs.C_GRID:
            rows.append(
                {"eta_rt": eta, "c": c, "delta_forecast": 1.0}
            )

    grid = pd.DataFrame(rows)
    verdict, n_positive, primary = rfhs.classify_section_8(grid)
    assert verdict == "CONFIRMATION"
    assert n_positive == 9
    assert primary > 0

    grid.loc[0, "delta_forecast"] = -1.0
    verdict, n_positive, _ = rfhs.classify_section_8(grid)
    assert verdict == "PARTIAL CONFIRMATION"
    assert n_positive == 8

    primary_mask = (
        (grid["eta_rt"] == rfhs.PRIMARY_ETA_RT)
        & (grid["c"] == rfhs.PRIMARY_C)
    )
    grid.loc[primary_mask, "delta_forecast"] = -0.1
    verdict, _, primary = rfhs.classify_section_8(grid)
    assert verdict == "FAILURE"
    assert primary <= 0


def test_full_end_to_end_sensitivity_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rfhs, "REPO_ROOT", tmp_path)
    _write_holdout_artifacts(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_final_holdout_sensitivity.py", "holdout_sensitivity_v1"],
    )
    rfhs.main()

    out_dir = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfhs.INPUT_STEM
        / "holdout_sensitivity_v1"
    )
    assert (out_dir / "sensitivity_grid.csv").exists()
    assert (out_dir / "per_day_sensitivity_results.csv").exists()
    assert (out_dir / "sensitivity_manifest.json").exists()

    grid = pd.read_csv(out_dir / "sensitivity_grid.csv")
    assert len(grid) == 9
    assert not grid[["eta_rt", "c"]].duplicated().any()

    detailed = pd.read_csv(out_dir / "per_day_sensitivity_results.csv")
    assert len(detailed) == 9 * 2

    manifest = json.loads(
        (out_dir / "sensitivity_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["base_case_reproduction"] == "PASSED"
    assert manifest["classification_complete"] is True
    assert manifest["holdout_used"] is True
    assert manifest["n_grid_cells"] == 9
    assert manifest["section_8_verdict"] in {
        "CONFIRMATION",
        "PARTIAL CONFIRMATION",
        "FAILURE",
    }


def test_output_overwrite_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rfhs, "REPO_ROOT", tmp_path)
    _write_holdout_artifacts(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_final_holdout_sensitivity.py", "holdout_sensitivity_v1"],
    )
    rfhs.main()
    with pytest.raises(FileExistsError):
        rfhs.main()
