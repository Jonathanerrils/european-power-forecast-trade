"""Tests for run_final_holdout_tailrisk.py.

No real 2026 artifacts are touched. Tests cover:
  - frozen source lineage;
  - holdout date/P&L validation;
  - inherited 95%/99% empirical VaR/ES method;
  - low-precision flags;
  - read-only output generation;
  - overwrite protection.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_final_holdout_tailrisk as rfht
from run_strategy_backtest import STRATEGIES


def test_resolve_run_args_accepts_only_output_version():
    assert (
        rfht.resolve_run_args(["holdout_tailrisk_v1"])
        == "holdout_tailrisk_v1"
    )
    with pytest.raises(SystemExit):
        rfht.resolve_run_args([])
    with pytest.raises(SystemExit):
        rfht.resolve_run_args(["holdout_v1", "holdout_tailrisk_v1"])


def test_frozen_risk_contract_is_inherited():
    assert rfht.HOLDOUT_RUN_VERSION == "holdout_v1"
    assert rfht.CONFIDENCE_LEVELS == [0.95, 0.99]
    assert rfht.LOW_PRECISION_TAIL_THRESHOLD == 20
    assert rfht.PRIMARY_ETA_RT == 0.85
    assert rfht.PRIMARY_C == 10.0


def _synthetic_per_day(n_days=30):
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
    rows = []
    for i, day in enumerate(dates):
        row = {
            "delivery_date": day.strftime("%Y-%m-%d"),
            "n_hours": 24,
        }
        for j, s in enumerate(STRATEGIES):
            if s == "S0":
                pnl = 0.0
            else:
                # Profitable overall with deterministic downside tail.
                pnl = 15.0 + j - (35.0 if i % 11 == 0 else 0.0)
            row[f"{s}_net_pnl"] = pnl
        rows.append(row)
    return pd.DataFrame(rows)


def _write_holdout_pnl(
    tmp_path,
    *,
    bad_protocol=False,
    holdout_used=True,
    per_day=None,
):
    holdout_dir = (
        tmp_path
        / "outputs"
        / "holdout"
        / rfht.INPUT_STEM
        / rfht.HOLDOUT_RUN_VERSION
    )
    holdout_dir.mkdir(parents=True)

    if per_day is None:
        per_day = _synthetic_per_day()

    per_day.to_csv(holdout_dir / "per_day_results.csv", index=False)

    manifest = {
        "output_run_version": rfht.HOLDOUT_RUN_VERSION,
        "holdout_used": holdout_used,
        "classification_complete": False,
        "holdout_start_local": rfht.HOLDOUT_START_LOCAL,
        "holdout_end_local_exclusive": rfht.HOLDOUT_END_LOCAL_EXCLUSIVE,
        "eta_rt_primary": rfht.PRIMARY_ETA_RT,
        "c_primary": rfht.PRIMARY_C,
        "strategies": list(STRATEGIES),
        "structural_invariants": "PASSED",
        "exact_raw_holdout_coverage": "PASSED",
        "protocol_tag": rfht.PROTOCOL_TAG,
        "protocol_commit": "wrong" if bad_protocol else rfht.PROTOCOL_COMMIT,
        "final_model_fit_tag": rfht.FINAL_MODEL_FIT_TAG,
        "final_model_fit_commit": rfht.FINAL_MODEL_FIT_COMMIT,
        "n_common_holdout_days": len(per_day),
    }
    (holdout_dir / "holdout_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return per_day


def test_load_holdout_pnl_accepts_valid_source(tmp_path, monkeypatch):
    monkeypatch.setattr(rfht, "REPO_ROOT", tmp_path)
    per_day = _write_holdout_pnl(tmp_path)

    loaded, manifest, _ = rfht.load_holdout_pnl()
    assert len(loaded) == len(per_day)
    assert manifest["holdout_used"] is True


def test_load_holdout_pnl_rejects_wrong_protocol(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rfht, "REPO_ROOT", tmp_path)
    _write_holdout_pnl(tmp_path, bad_protocol=True)

    with pytest.raises(
        ValueError, match="HOLDOUT TAIL-RISK SOURCE CHECK FAILED"
    ):
        rfht.load_holdout_pnl()


def test_load_holdout_pnl_rejects_holdout_used_false(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rfht, "REPO_ROOT", tmp_path)
    _write_holdout_pnl(tmp_path, holdout_used=False)

    with pytest.raises(
        ValueError, match="HOLDOUT TAIL-RISK SOURCE CHECK FAILED"
    ):
        rfht.load_holdout_pnl()


def test_validate_holdout_pnl_accepts_valid_data():
    per_day = _synthetic_per_day()
    manifest = {"n_common_holdout_days": len(per_day)}
    rfht.validate_holdout_pnl(per_day, manifest)


def test_validate_holdout_pnl_rejects_date_outside_window():
    per_day = _synthetic_per_day()
    per_day.loc[0, "delivery_date"] = "2025-12-31"
    manifest = {"n_common_holdout_days": len(per_day)}

    with pytest.raises(
        ValueError, match="outside frozen holdout window"
    ):
        rfht.validate_holdout_pnl(per_day, manifest)


def test_validate_holdout_pnl_rejects_duplicate_day():
    per_day = _synthetic_per_day()
    per_day.loc[1, "delivery_date"] = per_day.loc[0, "delivery_date"]
    manifest = {"n_common_holdout_days": len(per_day)}

    with pytest.raises(
        ValueError, match="duplicate day"
    ):
        rfht.validate_holdout_pnl(per_day, manifest)


def test_validate_holdout_pnl_rejects_nonzero_s0():
    per_day = _synthetic_per_day()
    per_day.loc[0, "S0_net_pnl"] = 1.0
    manifest = {"n_common_holdout_days": len(per_day)}

    with pytest.raises(
        ValueError, match="S0_net_pnl must be exactly zero"
    ):
        rfht.validate_holdout_pnl(per_day, manifest)


def test_validate_holdout_pnl_rejects_nonfinite_pnl():
    per_day = _synthetic_per_day()
    per_day.loc[0, "S2_net_pnl"] = np.nan
    manifest = {"n_common_holdout_days": len(per_day)}

    with pytest.raises(
        ValueError, match="S2_net_pnl contains"
    ):
        rfht.validate_holdout_pnl(per_day, manifest)


def test_tailrisk_uses_same_loss_sign_and_empirical_method():
    per_day = _synthetic_per_day(n_days=100)
    loss = rfht.compute_loss_series(per_day, "S2")

    np.testing.assert_allclose(
        loss.to_numpy(),
        -per_day["S2_net_pnl"].to_numpy(),
    )

    r95 = rfht.compute_empirical_var_es(loss, 0.95)
    r99 = rfht.compute_empirical_var_es(loss, 0.99)

    assert r95["confidence_level"] == 0.95
    assert r99["confidence_level"] == 0.99
    assert r95["n_total"] == 100
    assert r95["n_tail"] == pytest.approx(5.0)
    assert r99["n_tail"] == pytest.approx(1.0)
    assert r95["low_precision"] is True
    assert r99["low_precision"] is True
    assert r95["es"] >= r95["var"]
    assert r99["es"] >= r99["var"]


def test_full_end_to_end_tailrisk_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rfht, "REPO_ROOT", tmp_path)
    per_day = _write_holdout_pnl(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_final_holdout_tailrisk.py", "holdout_tailrisk_v1"],
    )
    rfht.main()

    out_dir = (
        tmp_path
        / "outputs"
        / "risk"
        / rfht.INPUT_STEM
        / "holdout_tailrisk_v1"
    )
    assert (out_dir / "var_es_results.csv").exists()
    assert (out_dir / "tailrisk_descriptive.csv").exists()
    assert (out_dir / "tailrisk_manifest.json").exists()

    results = pd.read_csv(out_dir / "var_es_results.csv")
    assert len(results) == len(STRATEGIES) * 2
    assert set(results["confidence_level"]) == {0.95, 0.99}
    assert set(results["strategy"]) == set(STRATEGIES)

    manifest = json.loads(
        (out_dir / "tailrisk_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["holdout_used"] is True
    assert manifest["decision_recomputation_performed"] is False
    assert manifest["forecast_recomputation_performed"] is False
    assert manifest["n_common_holdout_days"] == len(per_day)
    assert manifest["confidence_levels"] == [0.95, 0.99]


def test_tailrisk_output_overwrite_rejected(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rfht, "REPO_ROOT", tmp_path)
    _write_holdout_pnl(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_final_holdout_tailrisk.py", "holdout_tailrisk_v1"],
    )
    rfht.main()

    with pytest.raises(FileExistsError):
        rfht.main()
