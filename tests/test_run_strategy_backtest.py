"""Tests for run_strategy_backtest.py -- specifically the
common_strategy_evaluation_days construction (test 7 from
docs/economic_contract_v1.md) and the coverage/exclusion report, since
these are genuinely new orchestration logic not already covered by
test_strategy.py/test_oracle.py's pure-function tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_strategy_backtest as rsb


def _build_synthetic_day(date_str, n_hours, complete=True, missing_col=None):
    from src.clean import local_delivery_date_to_utc
    start = local_delivery_date_to_utc(date_str)  # correct LOCAL delivery-day boundary, not naive UTC midnight
    ts = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    rng = np.random.default_rng(hash(date_str) % (2**32))
    df = pd.DataFrame({
        "timestamp_utc": ts,
        "price_eur_mwh": rng.uniform(20, 100, n_hours),
        "lag_24_pred": rng.uniform(20, 100, n_hours),
        "xgboost_full_pred": rng.uniform(20, 100, n_hours),
        "xgboost_tier1_pred": rng.uniform(20, 100, n_hours),
        "full_L": rng.uniform(0, 20, n_hours),
        "full_U": rng.uniform(80, 120, n_hours),
        "tier1_L": rng.uniform(0, 20, n_hours),
        "tier1_U": rng.uniform(80, 120, n_hours),
    })
    if not complete and missing_col is not None:
        df.loc[df.index[0], missing_col] = np.nan  # one missing hour
    return df


# ---------------------------------------------------------------------
# build_common_evaluation_days
# ---------------------------------------------------------------------
def test_complete_days_are_all_included():
    day1 = _build_synthetic_day("2023-06-01", 24)
    day2 = _build_synthetic_day("2023-06-02", 24)
    combined = pd.concat([day1, day2], ignore_index=True)
    result = rsb.build_common_evaluation_days(combined)
    assert result["coverage"]["raw_delivery_days"] == 2
    assert result["coverage"]["common_evaluation_days"] == 2
    assert len(result["common_days"]) == 2


def test_a_day_with_one_missing_hour_is_excluded():
    """The core property this function exists for: even ONE missing
    hour of ONE required column makes the whole day ineligible -- a
    day-level decision needs a complete intraday candidate-pair search.
    """
    complete_day = _build_synthetic_day("2023-06-01", 24, complete=True)
    incomplete_day = _build_synthetic_day("2023-06-02", 24, complete=False, missing_col="xgboost_full_pred")
    combined = pd.concat([complete_day, incomplete_day], ignore_index=True)
    result = rsb.build_common_evaluation_days(combined)
    assert result["coverage"]["raw_delivery_days"] == 2
    assert result["coverage"]["common_evaluation_days"] == 1
    assert result["coverage"]["days_excluded_for_incomplete_intraday_vector"] == 1
    assert result["coverage"]["days_full_available"] == 1  # only the complete day


def test_missing_uncertainty_bounds_excludes_the_day_even_with_complete_point_forecasts():
    """A day where every point forecast exists but the uncertainty
    bounds are missing (e.g. still in the rolling window's warm-up
    period) must be excluded -- S3/S5 cannot be evaluated without L/U,
    and per the common-comparison-day rule, that means NO strategy is
    evaluated on that day either, not just S3/S5.
    """
    day = _build_synthetic_day("2023-01-05", 24, complete=False, missing_col="full_L")
    result = rsb.build_common_evaluation_days(day)
    assert result["coverage"]["common_evaluation_days"] == 0
    assert result["coverage"]["days_full_available"] == 1  # point forecast itself was complete
    assert result["coverage"]["days_uncertainty_full_available"] == 0


def test_dst_days_are_grouped_correctly_23_and_25_hours():
    """Spring-forward (23 local hours) and fall-back (25 local hours)
    days must each form exactly ONE delivery day, not be mis-split by
    a naive UTC-hour grouping.
    """
    spring = _build_synthetic_day("2024-03-31", 23)  # 2024-03-31 is spring DST in Berlin
    result = rsb.build_common_evaluation_days(spring)
    assert result["coverage"]["raw_delivery_days"] == 1

    fallback = _build_synthetic_day("2024-10-27", 25)  # 2024-10-27 is fall-back DST in Berlin
    result2 = rsb.build_common_evaluation_days(fallback)
    assert result2["coverage"]["raw_delivery_days"] == 1


# ---------------------------------------------------------------------
# run_backtest_for_day -- integration of strategy.py + oracle.py at the
# day-orchestration level
# ---------------------------------------------------------------------
def test_run_backtest_for_day_returns_all_six_strategies_and_oracle():
    day = _build_synthetic_day("2023-06-01", 24)
    day = day.sort_values("timestamp_utc").reset_index(drop=True)
    result = rsb.run_backtest_for_day(day, eta_rt=0.85, c=10.0)
    assert set(result.keys()) == {"S0", "S1", "S2", "S3", "S4", "S5", "oracle"}
    assert result["S0"]["net_pnl"] == 0.0
    assert result["S0"]["traded"] is False


def test_run_backtest_for_day_respects_oracle_dominance():
    rng = np.random.default_rng(0)
    for trial in range(20):
        day = _build_synthetic_day(f"2023-06-{trial+1:02d}", 24)
        day = day.sort_values("timestamp_utc").reset_index(drop=True)
        result = rsb.run_backtest_for_day(day, eta_rt=0.85, c=10.0)
        for strat in rsb.STRATEGIES:
            assert result[strat]["net_pnl"] <= result["oracle"] + 1e-6


# ---------------------------------------------------------------------
# verify_structural_invariants -- must actually catch a real violation
# ---------------------------------------------------------------------
def test_verify_structural_invariants_catches_dominance_violation():
    fake_results = [{
        "delivery_date": "2023-06-01",
        "oracle": 5.0,
        "S0": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S1": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S2": {"traded": True, "i": 2, "j": 5, "net_pnl": 999.0},  # impossibly exceeds oracle
        "S3": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S4": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S5": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
    }]
    with pytest.raises(AssertionError, match="structural invariant violation"):
        rsb.verify_structural_invariants(fake_results)


def test_verify_structural_invariants_catches_reverse_time():
    fake_results = [{
        "delivery_date": "2023-06-01",
        "oracle": 5.0,
        "S0": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S1": {"traded": True, "i": 5, "j": 2, "net_pnl": 1.0},  # i > j, invalid
        "S2": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S3": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S4": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S5": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
    }]
    with pytest.raises(AssertionError, match="structural invariant violation"):
        rsb.verify_structural_invariants(fake_results)


def test_verify_structural_invariants_passes_on_clean_results():
    fake_results = [{
        "delivery_date": "2023-06-01",
        "oracle": 5.0,
        "S0": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S1": {"traded": True, "i": 2, "j": 5, "net_pnl": 3.0},
        "S2": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S3": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S4": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
        "S5": {"traded": False, "i": None, "j": None, "net_pnl": 0.0},
    }]
    rsb.verify_structural_invariants(fake_results)  # must not raise


# ---------------------------------------------------------------------
# resolve_run_args
# ---------------------------------------------------------------------
def test_resolve_run_args_requires_exactly_four():
    with pytest.raises(SystemExit):
        rsb.resolve_run_args(["a", "b", "c"])
    with pytest.raises(SystemExit):
        rsb.resolve_run_args(["a", "b", "c", "d", "e", "f"])


def test_resolve_run_args_valid_call():
    result = rsb.resolve_run_args(["xg_v1", "unc_v1", "unc_tier1_v1", "out_v1"])
    assert result == ("xg_v1", "unc_v1", "unc_tier1_v1", "out_v1", None)


def test_resolve_run_args_with_optional_legacy_version():
    result = rsb.resolve_run_args(["xg_v1", "unc_v1", "unc_tier1_v1", "out_v2", "out_v1"])
    assert result == ("xg_v1", "unc_v1", "unc_tier1_v1", "out_v2", "out_v1")


# ---------------------------------------------------------------------
# verify_legacy_schema_extension_reproduction -- the v1->v2 bridge
# ---------------------------------------------------------------------
def _legacy_shaped_row(net_pnls: dict) -> dict:
    """Shaped like a LEGACY run's saved row -- no i/j columns, since
    those didn't exist before the schema-extension fix.
    """
    row = {"delivery_date": "2023-06-01", "n_hours": 24, "oracle_pnl": 100.0}
    for strat in rsb.STRATEGIES:
        row[f"{strat}_traded"] = True
        row[f"{strat}_net_pnl"] = net_pnls.get(strat, 0.0)
        row[f"{strat}_gross_pnl"] = net_pnls.get(strat, 0.0) + 5.0
    return row


def _new_shaped_row(net_pnls: dict) -> dict:
    """Shaped like the NEW run's saved row -- includes i/j, which the
    legacy comparison must correctly ignore (never attempt to compare
    a column the legacy run never had).
    """
    row = _legacy_shaped_row(net_pnls)
    for strat in rsb.STRATEGIES:
        row[f"{strat}_i"] = 2
        row[f"{strat}_j"] = 5
    return row


def test_legacy_reproduction_passes_on_genuine_schema_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(rsb, "REPO_ROOT", tmp_path)
    legacy_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "v1"
    legacy_dir.mkdir(parents=True)
    net_pnls = {"S0": 0.0, "S1": 10.0, "S2": 15.0, "S3": 5.0, "S4": 8.0, "S5": 2.0}
    pd.DataFrame([_legacy_shaped_row(net_pnls)]).to_csv(legacy_dir / "per_day_results.csv", index=False)

    new_results = pd.DataFrame([_new_shaped_row(net_pnls)])  # same P&L, plus new i/j columns
    result = rsb.verify_legacy_schema_extension_reproduction(new_results, "v1", "delu_features")
    assert result["parent_run"] == "v1"


def test_legacy_reproduction_catches_a_real_pnl_mismatch(tmp_path, monkeypatch):
    """The core purpose of the bridge: a code change that LOOKS purely
    additive but actually altered a P&L value must be caught, not
    assumed safe.
    """
    monkeypatch.setattr(rsb, "REPO_ROOT", tmp_path)
    legacy_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "v1"
    legacy_dir.mkdir(parents=True)
    legacy_pnls = {"S0": 0.0, "S1": 10.0, "S2": 15.0, "S3": 5.0, "S4": 8.0, "S5": 2.0}
    pd.DataFrame([_legacy_shaped_row(legacy_pnls)]).to_csv(legacy_dir / "per_day_results.csv", index=False)

    new_pnls = dict(legacy_pnls)
    new_pnls["S3"] = 5.5  # deliberately different -- something DID change
    new_results = pd.DataFrame([_new_shaped_row(new_pnls)])

    with pytest.raises(AssertionError, match="LEGACY SCHEMA-EXTENSION REPRODUCTION FAILED"):
        rsb.verify_legacy_schema_extension_reproduction(new_results, "v1", "delu_features")


def test_legacy_reproduction_raises_clearly_when_legacy_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(rsb, "REPO_ROOT", tmp_path)
    new_results = pd.DataFrame([_new_shaped_row({"S1": 10.0})])
    with pytest.raises(FileNotFoundError, match="No legacy results"):
        rsb.verify_legacy_schema_extension_reproduction(new_results, "never_run", "delu_features")


def test_legacy_reproduction_ignores_new_only_ij_columns_correctly(tmp_path, monkeypatch):
    """The comparison must never attempt to compare i/j against the
    legacy run -- those columns don't exist there. A run with
    COMPLETELY DIFFERENT (i, j) values but IDENTICAL P&L must still
    pass, since P&L (not the specific pair) is what the legacy run
    actually recorded.
    """
    monkeypatch.setattr(rsb, "REPO_ROOT", tmp_path)
    legacy_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "v1"
    legacy_dir.mkdir(parents=True)
    net_pnls = {"S0": 0.0, "S1": 10.0, "S2": 15.0, "S3": 5.0, "S4": 8.0, "S5": 2.0}
    pd.DataFrame([_legacy_shaped_row(net_pnls)]).to_csv(legacy_dir / "per_day_results.csv", index=False)

    new_row = _new_shaped_row(net_pnls)
    for strat in rsb.STRATEGIES:
        new_row[f"{strat}_i"], new_row[f"{strat}_j"] = 7, 11  # deliberately different pair
    new_results = pd.DataFrame([new_row])
    rsb.verify_legacy_schema_extension_reproduction(new_results, "v1", "delu_features")  # must not raise


# ---------------------------------------------------------------------
# main()-level test: the ordering guarantee itself, not just the
# comparison function in isolation. Proves the specific claim a design
# review required: on a legacy mismatch, this run's own economic
# figures ("SUMMARY", "Delta_forecast") must NEVER reach stdout, and
# the output directory must remain absent -- not just that the
# function eventually raises.
# ---------------------------------------------------------------------
def _write_synthetic_fold_predictions(xgb_dir):
    from src.clean import local_delivery_date_to_utc
    rng = np.random.default_rng(0)
    fold_windows = [
        ("fold_1", "2023-01-01", "2024-01-01"), ("fold_2", "2024-01-01", "2025-01-01"),
        ("fold_3", "2025-01-01", "2025-10-01"), ("regime_stress_test", "2025-10-01", "2026-01-01"),
    ]
    for name, start_str, end_str in fold_windows:
        start, end = local_delivery_date_to_utc(start_str), local_delivery_date_to_utc(end_str)
        ts = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
        price = 80 + 20 * np.sin(np.arange(len(ts)) / 24) + rng.normal(0, 15, len(ts))
        df = pd.DataFrame({
            "timestamp_utc": ts, "price_eur_mwh": price,
            "lag_24_pred": price + rng.normal(0, 10, len(ts)),
            "xgboost_full_pred": price + rng.normal(0, 5, len(ts)),
            "xgboost_tier1_pred": price + rng.normal(0, 12, len(ts)),
        })
        df.to_csv(xgb_dir / f"{name}_predictions.csv", index=False)


def test_main_never_prints_economics_before_a_failed_legacy_gate(tmp_path, monkeypatch, capsys):
    """Full main()-level proof: builds a real legacy artifact, corrupts
    ONE P&L value in it (simulating a legacy run that a genuine
    schema-extension would NOT actually reproduce), then confirms
    main() raises WITHOUT ever printing "SUMMARY" or "Delta_forecast"
    to stdout, and without creating the output directory.
    """
    import run_uncertainty as ru
    import run_uncertainty_tier1_robustness as rt1

    xgb_dir = tmp_path / "outputs" / "models" / "delu_features" / "xgboost_v1_a03fix"
    xgb_dir.mkdir(parents=True)
    _write_synthetic_fold_predictions(xgb_dir)

    for mod in (ru, rt1, rsb):
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)

    monkeypatch.setattr(sys, "argv", ["run_uncertainty.py", "xgboost_v1_a03fix", "uncertainty_selected_v1", "60"])
    ru.main()
    monkeypatch.setattr(sys, "argv", ["run_uncertainty_tier1_robustness.py", "xgboost_v1_a03fix", "unc_tier1_v1"])
    rt1.main()

    # Build a genuine legacy artifact, then corrupt it -- this simulates
    # exactly the scenario the review was worried about: a legacy run
    # that a supposedly schema-only change does NOT actually reproduce.
    monkeypatch.setattr(sys, "argv", [
        "run_strategy_backtest.py", "xgboost_v1_a03fix", "uncertainty_selected_v1", "unc_tier1_v1", "legacy_v1",
    ])
    rsb.main()
    capsys.readouterr()  # discard this first run's own output

    legacy_path = tmp_path / "outputs" / "strategy" / "delu_features" / "legacy_v1" / "per_day_results.csv"
    legacy_df = pd.read_csv(legacy_path)
    legacy_df.loc[0, "S2_net_pnl"] += 500.0  # deliberate corruption -- something DID change
    legacy_df.to_csv(legacy_path, index=False)

    monkeypatch.setattr(sys, "argv", [
        "run_strategy_backtest.py", "xgboost_v1_a03fix", "uncertainty_selected_v1", "unc_tier1_v1",
        "new_v2", "legacy_v1",
    ])
    with pytest.raises(AssertionError, match="LEGACY SCHEMA-EXTENSION REPRODUCTION FAILED"):
        rsb.main()

    captured = capsys.readouterr()
    assert "SUMMARY" not in captured.out, "Economic summary was printed despite a failed legacy gate"
    assert "Delta_forecast" not in captured.out, "Deltas were printed despite a failed legacy gate"

    new_out_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "new_v2"
    assert not new_out_dir.exists() or not any(new_out_dir.iterdir()), (
        "Output directory was created/populated despite a failed legacy gate"
    )
