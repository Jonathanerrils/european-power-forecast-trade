"""Tests for run_strategy_decomposition.py -- the reconciliation
assertion is the most important thing here, tested both on a genuine
pass and on constructed failures.
"""
import sys
from pathlib import Path
import json

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_strategy_decomposition as rsd


def _make_day(delivery_date, s2_traded, s2_i, s2_j, s2_pnl, s3_traded, s3_i, s3_j, s3_pnl):
    return {
        "delivery_date": delivery_date,
        "S2_traded": s2_traded, "S2_i": s2_i, "S2_j": s2_j, "S2_net_pnl": s2_pnl,
        "S3_traded": s3_traded, "S3_i": s3_i, "S3_j": s3_j, "S3_net_pnl": s3_pnl,
        "S4_traded": False, "S4_i": None, "S4_j": None, "S4_net_pnl": 0.0,
        "S5_traded": False, "S5_i": None, "S5_j": None, "S5_net_pnl": 0.0,
    }


# ---------------------------------------------------------------------
# classify_day
# ---------------------------------------------------------------------
def test_classify_both_abstain():
    row = pd.Series(_make_day("d", False, None, None, 0.0, False, None, None, 0.0))
    assert rsd.classify_day(row, "S2", "S3") == "both_abstain"


def test_classify_point_trades_uncertainty_abstains():
    row = pd.Series(_make_day("d", True, 2, 5, 10.0, False, None, None, 0.0))
    assert rsd.classify_day(row, "S2", "S3") == "point_trades_uncertainty_abstains"


def test_classify_point_abstains_uncertainty_trades():
    row = pd.Series(_make_day("d", False, None, None, 0.0, True, 2, 5, 10.0))
    assert rsd.classify_day(row, "S2", "S3") == "point_abstains_uncertainty_trades"


def test_classify_both_trade_same_pair():
    row = pd.Series(_make_day("d", True, 2, 5, 10.0, True, 2, 5, 10.0))
    assert rsd.classify_day(row, "S2", "S3") == "both_trade_same_pair"


def test_classify_both_trade_different_pair():
    row = pd.Series(_make_day("d", True, 2, 5, 10.0, True, 3, 6, 8.0))
    assert rsd.classify_day(row, "S2", "S3") == "both_trade_different_pair"


# ---------------------------------------------------------------------
# decompose -- the reconciliation assertion, tested both ways
# ---------------------------------------------------------------------
def test_decompose_reconciles_exactly_on_a_constructed_case():
    """Hand-verifiable by inspection: profits forgone = 20 (day 2's
    positive S2 P&L that S3 missed), losses avoided = 5 (day 3's
    negative S2 P&L that S3 correctly avoided), pair-selection effect
    = -3 (day 5's S3-minus-S2 on a different-pair day), same-pair
    days contribute 0. Total gap must equal -20 + 5 + (-3) = -18.
    """
    days = [
        _make_day("d1", False, None, None, 0.0, False, None, None, 0.0),   # both abstain
        _make_day("d2", True, 1, 2, 20.0, False, None, None, 0.0),          # forgone profit of 20
        _make_day("d3", True, 1, 2, -5.0, False, None, None, 0.0),          # avoided loss of 5
        _make_day("d4", True, 1, 2, 10.0, True, 1, 2, 10.0),                # same pair, contributes 0
        _make_day("d5", True, 1, 2, 15.0, True, 3, 4, 12.0),                # different pair, effect = 12-15 = -3
    ]
    df = pd.DataFrame(days)
    result = rsd.decompose(df, "S2", "S3")

    assert result["profits_forgone"] == pytest.approx(20.0)
    assert result["losses_avoided"] == pytest.approx(5.0)
    assert result["pair_selection_effect"] == pytest.approx(-3.0)
    # Hand-verified per day: d1=0 (both abstain), d2=0-20=-20, d3=0-(-5)=5,
    # d4=10-10=0 (same pair), d5=12-15=-3 (different pair). Total = -18.
    expected_total_gap = 0 + (0 - 20) + (0 - (-5)) + (10 - 10) + (12 - 15)
    assert expected_total_gap == -18.0  # sanity-check the hand calculation itself
    assert result["total_gap"] == pytest.approx(expected_total_gap)
    assert result["reconciliation"] == "EXACT"


def test_decompose_raises_on_same_pair_pnl_mismatch():
    """Structural violation: identical (i, j) on the same day with the
    same actual prices realizes identical P&L, regardless of which
    rule chose it -- if two 'same pair' P&Ls differ, something in the
    upstream backtest is broken, and the decomposition must refuse to
    silently absorb that into the reconciliation.
    """
    days = [_make_day("d1", True, 2, 5, 10.0, True, 2, 5, 12.0)]  # same pair, DIFFERENT P&L
    df = pd.DataFrame(days)
    with pytest.raises(AssertionError, match="STRUCTURAL VIOLATION"):
        rsd.decompose(df, "S2", "S3")


def test_decompose_reports_reverse_category_as_diagnostic_not_hard_failure():
    """point_abstains_uncertainty_trades is expected to be rare/empty
    in practice but is NOT asserted to be exactly zero (it's an
    empirical property of the real residual distribution, not a
    mathematical guarantee) -- confirmed here that a nonzero count is
    reported, not raised.
    """
    days = [_make_day("d1", False, None, None, 0.0, True, 2, 5, 7.0)]
    df = pd.DataFrame(days)
    result = rsd.decompose(df, "S2", "S3")  # must not raise
    assert result["n_point_abstains_uncertainty_trades"] == 1
    assert result["diagnostic_reverse_category_nonzero"] is True


def test_decompose_handles_empty_categories_gracefully():
    days = [_make_day("d1", False, None, None, 0.0, False, None, None, 0.0)]
    df = pd.DataFrame(days)
    result = rsd.decompose(df, "S2", "S3")
    assert result["profits_forgone"] == 0.0
    assert result["losses_avoided"] == 0.0
    assert result["pair_selection_effect"] == 0.0
    assert result["total_gap"] == 0.0


def test_decompose_matches_a_realistic_scale_reconciliation():
    """A larger, randomized case -- confirms reconciliation holds
    exactly (not just approximately) at realistic day counts, not only
    the small hand-verified example above.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    days = []
    for d in range(500):
        s2_traded = bool(rng.random() < 0.9)
        s3_traded = bool(rng.random() < 0.6) if s2_traded else False  # S3 subset of S2, roughly
        s2_pnl = float(rng.normal(20, 50)) if s2_traded else 0.0
        if s3_traded:
            same_pair = rng.random() < 0.5
            if same_pair:
                s3_pnl, s3_i, s3_j = s2_pnl, 1, 2
            else:
                s3_pnl, s3_i, s3_j = float(rng.normal(15, 40)), 3, 4
        else:
            s3_pnl, s3_i, s3_j = 0.0, None, None
        days.append(_make_day(f"d{d}", s2_traded, 1 if s2_traded else None, 2 if s2_traded else None,
                               s2_pnl, s3_traded, s3_i, s3_j, s3_pnl))
    df = pd.DataFrame(days)
    result = rsd.decompose(df, "S2", "S3")  # must not raise -- reconciliation holds by construction
    assert result["reconciliation"] == "EXACT"


# ---------------------------------------------------------------------
# resolve_run_args / load_backtest_results
# ---------------------------------------------------------------------
def test_resolve_run_args_requires_exactly_two():
    with pytest.raises(SystemExit):
        rsd.resolve_run_args(["a"])
    with pytest.raises(SystemExit):
        rsd.resolve_run_args(["a", "b", "c"])


def test_load_backtest_results_raises_clearly_when_missing_ij(tmp_path, monkeypatch):
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "old_v1"
    run_dir.mkdir(parents=True)
    old_format = pd.DataFrame([{
        "delivery_date": "2023-06-01", "S2_traded": True, "S2_net_pnl": 10.0,
        "S3_traded": True, "S3_net_pnl": 8.0, "S4_traded": True, "S4_net_pnl": 5.0,
        "S5_traded": True, "S5_net_pnl": 3.0,
    }])
    old_format.to_csv(run_dir / "per_day_results.csv", index=False)
    with pytest.raises(ValueError, match="requires .i, j. persisted"):
        rsd.load_backtest_results("delu_features", "old_v1")


def test_load_backtest_results_raises_clearly_when_missing_entirely(tmp_path, monkeypatch):
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing"):
        rsd.load_backtest_results("delu_features", "never_run")


# ---------------------------------------------------------------------
# validate_backtest_results -- fail-closed on malformed persisted input.
# Each test corresponds to a specific failure mode a design review
# identified and I confirmed by direct reproduction before fixing:
# NaN P&L, non-boolean traded flags, and inconsistent traded/(i,j)/P&L
# combinations were all previously silently absorbed rather than
# rejected.
# ---------------------------------------------------------------------
def _valid_df(n=5):
    import numpy as np
    rows = []
    for d in range(n):
        rows.append({
            "delivery_date": f"2023-06-{d+1:02d}", "n_hours": 24,
            "S2_traded": True, "S2_i": 2, "S2_j": 5, "S2_net_pnl": 20.0,
            "S3_traded": False, "S3_i": np.nan, "S3_j": np.nan, "S3_net_pnl": 0.0,
            "S4_traded": True, "S4_i": 1, "S4_j": 4, "S4_net_pnl": 15.0,
            "S5_traded": False, "S5_i": np.nan, "S5_j": np.nan, "S5_net_pnl": 0.0,
        })
    return pd.DataFrame(rows)


def test_validate_passes_on_clean_data():
    df = _valid_df()
    rsd.validate_backtest_results(df)  # must not raise


def test_validate_catches_nan_net_pnl():
    import numpy as np
    df = _valid_df()
    df.loc[2, "S2_net_pnl"] = np.nan
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_inf_net_pnl():
    df = _valid_df()
    df.loc[2, "S2_net_pnl"] = float("inf")
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_nan_traded_flag():
    import numpy as np
    df = _valid_df()
    df["S2_traded"] = df["S2_traded"].astype(object)
    df.loc[2, "S2_traded"] = np.nan
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_string_traded_flag():
    df = _valid_df()
    df["S2_traded"] = df["S2_traded"].astype(object)
    df.loc[2, "S2_traded"] = "False"  # the exact bool("False")==True trap
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_traded_true_with_missing_ij():
    import numpy as np
    df = _valid_df()
    df.loc[2, "S2_i"] = np.nan
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_no_trade_with_present_ij():
    df = _valid_df()
    df.loc[2, "S3_i"] = 3
    df.loc[2, "S3_j"] = 7
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_no_trade_with_nonzero_pnl():
    df = _valid_df()
    df.loc[2, "S3_net_pnl"] = 5.0
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_duplicate_delivery_date():
    df = _valid_df()
    df.loc[1, "delivery_date"] = df.loc[0, "delivery_date"]
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_i_greater_equal_j():
    df = _valid_df()
    df.loc[2, "S2_i"] = 6
    df.loc[2, "S2_j"] = 3  # i > j, invalid
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_reports_multiple_problems_together():
    """Confirms the function collects ALL problems rather than stopping
    at the first one -- more useful for a real malformed artifact with
    several issues at once.
    """
    import numpy as np
    df = _valid_df()
    df.loc[2, "S2_net_pnl"] = np.nan
    df.loc[3, "S3_net_pnl"] = 5.0  # no-trade with nonzero pnl
    with pytest.raises(ValueError) as exc_info:
        rsd.validate_backtest_results(df)
    msg = str(exc_info.value)
    assert "S2_net_pnl" in msg
    assert "S3" in msg


def test_end_to_end_nan_pnl_no_longer_silently_reconciles():
    """The exact scenario a design review found: a NaN S2_net_pnl with
    S3 abstaining previously produced total_gap=0, reconciliation=EXACT
    -- silently masquerading a missing observation as a clean result.
    Confirms the fix: this now raises during validation, well before
    decompose() could ever silently absorb it.
    """
    import numpy as np
    row = {
        "delivery_date": "d1", "n_hours": 24,
        "S2_traded": True, "S2_i": 2, "S2_j": 5, "S2_net_pnl": np.nan,
        "S3_traded": False, "S3_i": None, "S3_j": None, "S3_net_pnl": 0.0,
        "S4_traded": False, "S4_i": None, "S4_j": None, "S4_net_pnl": 0.0,
        "S5_traded": False, "S5_i": None, "S5_j": None, "S5_net_pnl": 0.0,
    }
    df = pd.DataFrame([row])
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


# ---------------------------------------------------------------------
# verify_source_provenance
# ---------------------------------------------------------------------
def _write_manifest(run_dir, **overrides):
    import json
    manifest = {
        "eta_rt": 0.85, "c": 10.0, "holdout_used": False,
        "structural_invariants": "PASSED", "parent_run": "strategy_backtest_v1",
        "change_type": "schema_extension_only", "legacy_result_reproduction": "PASSED",
    }
    manifest.update(overrides)
    with open(run_dir / "strategy_backtest_manifest.json", "w") as f:
        json.dump(manifest, f)


def test_provenance_passes_on_genuine_primary_case(tmp_path, monkeypatch):
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "v2"
    run_dir.mkdir(parents=True)
    _write_manifest(run_dir)
    result = rsd.verify_source_provenance("v2", "delu_features")
    assert result["eta_rt"] == 0.85


def test_provenance_catches_wrong_eta(tmp_path, monkeypatch):
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "sensitivity_cell"
    run_dir.mkdir(parents=True)
    _write_manifest(run_dir, eta_rt=0.70)  # a sensitivity cell, not the primary case
    with pytest.raises(ValueError, match="SOURCE PROVENANCE CHECK FAILED"):
        rsd.verify_source_provenance("sensitivity_cell", "delu_features")


def test_provenance_catches_holdout_used(tmp_path, monkeypatch):
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "bad_run"
    run_dir.mkdir(parents=True)
    _write_manifest(run_dir, holdout_used=True)
    with pytest.raises(ValueError, match="SOURCE PROVENANCE CHECK FAILED"):
        rsd.verify_source_provenance("bad_run", "delu_features")


def test_provenance_catches_failed_structural_invariants(tmp_path, monkeypatch):
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "bad_run"
    run_dir.mkdir(parents=True)
    _write_manifest(run_dir, structural_invariants="FAILED")
    with pytest.raises(ValueError, match="SOURCE PROVENANCE CHECK FAILED"):
        rsd.verify_source_provenance("bad_run", "delu_features")


def test_provenance_raises_clearly_when_manifest_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing"):
        rsd.verify_source_provenance("never_run", "delu_features")


def test_provenance_notes_but_does_not_fail_when_no_parent_run(tmp_path, monkeypatch, capsys):
    """A genuinely first-ever run with no legacy predecessor is still
    valid -- the missing 'parent_run' lineage field is a printed note,
    not a hard failure.
    """
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "first_run"
    run_dir.mkdir(parents=True)
    _write_manifest(run_dir, parent_run=None)
    del_manifest = json.loads((run_dir / "strategy_backtest_manifest.json").read_text())
    del del_manifest["parent_run"]
    (run_dir / "strategy_backtest_manifest.json").write_text(json.dumps(del_manifest))

    result = rsd.verify_source_provenance("first_run", "delu_features")  # must not raise
    assert result["eta_rt"] == 0.85


# ---------------------------------------------------------------------
# Index integrality and bounds -- the final gap a design review found:
# a fractional or out-of-range (i, j) would pass a bare i<j check
# without corresponding to a real hourly delivery interval.
# ---------------------------------------------------------------------
def test_validate_catches_fractional_i():
    df = _valid_df()
    # Real malformed CSVs get float dtype for the WHOLE column, not a
    # single mutated cell -- .loc assignment onto a clean int64 column
    # would itself refuse the fractional value before validate_backtest_results
    # even runs, which wouldn't reproduce the actual real-world failure mode.
    df["S2_i"] = df["S2_i"].astype(float)
    df["S2_j"] = df["S2_j"].astype(float)
    df.loc[2, "S2_i"] = 2.5
    df.loc[2, "S2_j"] = 7.5
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_negative_i():
    df = _valid_df()
    df.loc[2, "S2_i"] = -1
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_j_out_of_bounds():
    df = _valid_df()  # n_hours=24 for every row
    df.loc[2, "S2_j"] = 999
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_accepts_j_at_the_valid_boundary():
    """j = n_hours - 1 is the last valid index (0-indexed) and must be
    accepted, not rejected by an off-by-one bounds error.
    """
    df = _valid_df()
    df.loc[2, "S2_i"] = 0
    df.loc[2, "S2_j"] = 23  # valid last hour for a 24-hour day
    rsd.validate_backtest_results(df)  # must not raise


def test_load_backtest_results_raises_clearly_when_delivery_date_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "bad_run"
    run_dir.mkdir(parents=True)
    df = _valid_df().drop(columns=["delivery_date"])
    df.to_csv(run_dir / "per_day_results.csv", index=False)
    with pytest.raises(ValueError, match=r"missing \['delivery_date'\]"):
        rsd.load_backtest_results("delu_features", "bad_run")


def test_load_backtest_results_raises_clearly_when_n_hours_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "bad_run"
    run_dir.mkdir(parents=True)
    df = _valid_df().drop(columns=["n_hours"])
    df.to_csv(run_dir / "per_day_results.csv", index=False)
    with pytest.raises(ValueError, match=r"missing \['n_hours'\]"):
        rsd.load_backtest_results("delu_features", "bad_run")


# ---------------------------------------------------------------------
# Provenance lineage tightening (optional per the review, implemented
# anyway since it was cheap and closes a real gap)
# ---------------------------------------------------------------------
def test_provenance_catches_parent_run_without_schema_extension_change_type(tmp_path, monkeypatch):
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "bad_run"
    run_dir.mkdir(parents=True)
    _write_manifest(run_dir, change_type="something_else")
    with pytest.raises(ValueError, match="SOURCE PROVENANCE CHECK FAILED"):
        rsd.verify_source_provenance("bad_run", "delu_features")


def test_provenance_catches_parent_run_without_passed_legacy_reproduction(tmp_path, monkeypatch):
    monkeypatch.setattr(rsd, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "bad_run"
    run_dir.mkdir(parents=True)
    _write_manifest(run_dir, legacy_result_reproduction="FAILED")
    with pytest.raises(ValueError, match="SOURCE PROVENANCE CHECK FAILED"):
        rsd.verify_source_provenance("bad_run", "delu_features")


# ---------------------------------------------------------------------
# Final fail-closed hardening: n_hours domain + non-numeric persisted i/j
# ---------------------------------------------------------------------
def test_validate_catches_fractional_n_hours():
    df = _valid_df()
    df["n_hours"] = df["n_hours"].astype(float)
    df.loc[2, "n_hours"] = 24.5
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_n_hours_outside_dst_domain():
    df = _valid_df()
    df.loc[2, "n_hours"] = 26
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_accepts_all_three_valid_dst_lengths():
    """23 (spring-forward), 24 (normal), and 25 (fall-back) must all
    be accepted -- confirms the domain check isn't accidentally
    over-restrictive to just 24.
    """
    for valid_n in (23, 24, 25):
        df = _valid_df(n=1)
        df.loc[0, "n_hours"] = valid_n
        df.loc[0, "S2_i"] = 0
        df.loc[0, "S2_j"] = valid_n - 1
        rsd.validate_backtest_results(df)  # must not raise


def test_validate_catches_nonnumeric_i_cleanly():
    """Confirms the fix for a real crash: a string in the i column
    previously raised an uncontrolled TypeError ("'>=' not supported
    between instances of 'str' and 'int'") from downstream arithmetic
    rather than a clean validation error.
    """
    df = _valid_df()
    df["S2_i"] = df["S2_i"].astype(object)
    df.loc[2, "S2_i"] = "not-an-index"
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)


def test_validate_catches_nonnumeric_j_cleanly():
    df = _valid_df()
    df["S2_j"] = df["S2_j"].astype(object)
    df.loc[2, "S2_j"] = "5x"
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rsd.validate_backtest_results(df)
