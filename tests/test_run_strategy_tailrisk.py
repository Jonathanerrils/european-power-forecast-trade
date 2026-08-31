"""Tests for run_strategy_tailrisk.py. Every VaR/ES case is hand-
verifiable by inspection -- no real backtest data involved for the
core math, matching this project's established discipline for new
quantitative layers.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_strategy_tailrisk as rst


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Architectural constraint: read-only, never touches decision logic
# ---------------------------------------------------------------------
def test_tailrisk_never_references_decision_making_functions():
    tree = ast.parse((REPO_ROOT / "run_strategy_tailrisk.py").read_text())
    forbidden = {"run_day", "best_point_pair", "best_uncertainty_pair", "realized_pnl", "oracle_pnl", "best_realized_pair"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            found.add(node.id)
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            found.add(node.attr)
    assert not found, f"run_strategy_tailrisk.py references decision-making function(s): {found}"


# ---------------------------------------------------------------------
# compute_loss_series -- L_D = -Pi_D, the ONLY place loss is defined
# ---------------------------------------------------------------------
def test_compute_loss_series_is_negated_pnl():
    from run_strategy_backtest import STRATEGIES
    df = pd.DataFrame([{f"{s}_net_pnl": v for s, v in zip(STRATEGIES, [0, 10, -5, 20, -3, 0])}])
    loss = rst.compute_loss_series(df, "S1")
    assert loss.iloc[0] == -10.0
    loss2 = rst.compute_loss_series(df, "S2")
    assert loss2.iloc[0] == 5.0  # a losing day (net_pnl=-5) becomes a POSITIVE loss


def test_loss_series_never_reuses_uncertainty_quantities():
    """Guards against the exact conflation this project's own docs
    explicitly warn against: L_D must come from realized strategy P&L,
    never from a price-forecast residual or an uncertainty interval
    width. Confirmed by checking the function's only data dependency
    is the *_net_pnl column, nothing uncertainty-related.
    """
    src = ast.parse((REPO_ROOT / "run_strategy_tailrisk.py").read_text())
    func = next(n for n in ast.walk(src) if isinstance(n, ast.FunctionDef) and n.name == "compute_loss_series")
    func_src = ast.dump(func)
    for forbidden_term in ("forecast_q10", "forecast_q90", "residual", "quantile_forecast"):
        assert forbidden_term not in func_src, f"compute_loss_series references '{forbidden_term}' -- loss must come from net_pnl only"


# ---------------------------------------------------------------------
# compute_empirical_var_es -- hand-verifiable by inspection
# ---------------------------------------------------------------------
def test_var_es_hand_verifiable_simple_case():
    """Losses: [1, 2, 3, ..., 100] (as a loss series, ascending).
    VaR_95 = 95th percentile = 95.05 (pandas linear interpolation).
    Tail = all values >= 95.05 = {96, 97, 98, 99, 100} -> n_tail=5, ES=mean=98.
    """
    losses = pd.Series(range(1, 101), dtype=float)  # 1..100
    result = rst.compute_empirical_var_es(losses, 0.95)
    assert result["n_total"] == 100
    assert result["var"] == pytest.approx(95.05, abs=0.5)
    assert result["n_tail"] == pytest.approx(5.0)
    assert result["es"] == pytest.approx(98.0)
    assert result["low_precision"] is True  # 5 < 20 threshold


def test_var_es_larger_tail_not_flagged_low_precision():
    losses = pd.Series(range(1, 1001), dtype=float)  # 1..1000
    result = rst.compute_empirical_var_es(losses, 0.95)
    assert result["n_tail"] >= 20
    assert result["low_precision"] is False


def test_var_es_all_identical_losses():
    """Degenerate case: every day has the same loss. VaR and ES must
    both equal that constant, not raise or produce NaN.
    """
    losses = pd.Series([10.0] * 50)
    result = rst.compute_empirical_var_es(losses, 0.95)
    assert result["var"] == pytest.approx(10.0)
    assert result["es"] == pytest.approx(10.0)


def test_var_es_rejects_invalid_confidence_level():
    losses = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="confidence_level must be"):
        rst.compute_empirical_var_es(losses, 1.5)
    with pytest.raises(ValueError, match="confidence_level must be"):
        rst.compute_empirical_var_es(losses, 0.0)


def test_var_es_rejects_empty_series():
    with pytest.raises(ValueError, match="empty loss series"):
        rst.compute_empirical_var_es(pd.Series([], dtype=float), 0.95)


def test_var_es_es_never_less_than_var():
    """ES is the mean of the tail (values >= VaR), so it can never be
    smaller than VaR itself for a well-formed loss distribution.
    """
    rng = np.random.default_rng(0)
    losses = pd.Series(rng.normal(0, 30, 500))
    for cl in [0.90, 0.95, 0.99]:
        result = rst.compute_empirical_var_es(losses, cl)
        assert result["es"] >= result["var"] - 1e-9


def test_var_es_matches_a_real_backtest_shaped_scenario():
    """Approximates the real S2 profile (mostly profitable, occasional
    losing days) -- confirms sensible, non-degenerate output on a
    realistic distribution shape, not just clean synthetic cases.
    """
    rng = np.random.default_rng(1)
    net_pnl = rng.normal(60, 40, 1081)  # profitable on average, like real S2
    loss = -pd.Series(net_pnl)
    result_95 = rst.compute_empirical_var_es(loss, 0.95)
    result_99 = rst.compute_empirical_var_es(loss, 0.99)
    assert result_99["var"] >= result_95["var"]  # 99% VaR must be at least as extreme as 95%
    assert result_99["low_precision"] is True   # ~11 tail obs at n=1081
    assert result_95["low_precision"] is False  # ~54 tail obs


# ---------------------------------------------------------------------
# resolve_run_args / load_backtest_pnl
# ---------------------------------------------------------------------
def test_resolve_run_args_requires_exactly_two():
    with pytest.raises(SystemExit):
        rst.resolve_run_args(["a"])
    with pytest.raises(SystemExit):
        rst.resolve_run_args(["a", "b", "c"])


def test_load_backtest_pnl_raises_clearly_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rst, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing"):
        rst.load_backtest_pnl("delu_features", "never_run")


def test_load_backtest_pnl_raises_clearly_on_missing_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(rst, "REPO_ROOT", tmp_path)
    run_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "bad_run"
    run_dir.mkdir(parents=True)
    pd.DataFrame([{"delivery_date": "d1"}]).to_csv(run_dir / "per_day_results.csv", index=False)
    with pytest.raises(ValueError, match="missing required column"):
        rst.load_backtest_pnl("delu_features", "bad_run")


# ---------------------------------------------------------------------
# Large point-mass at loss=0 -- THE bug a design review found: the
# tied-threshold ES method silently included hundreds of exact-zero
# abstention days as "tail" observations, diluting ES toward zero and
# making heavily-abstaining strategies (S3, S5) look far safer than
# their true worst-m-observations severity.
# ---------------------------------------------------------------------
def test_es_not_diluted_by_large_zero_point_mass():
    """Hand-verifiable: 10 large losses (100 each), 90 exact-zero
    (no-trade) days. n=100. At 95% confidence, m=5 -- the worst 5
    observations are all 100s, so ES_95 should be exactly 100, NOT
    diluted toward 0 by the 90 zero-tied observations.
    """
    losses = pd.Series([100.0] * 10 + [0.0] * 90)
    result = rst.compute_empirical_var_es(losses, 0.95)
    assert result["es"] == pytest.approx(100.0)
    assert result["n_tail"] == pytest.approx(5.0)


def test_es_matches_real_s3_shaped_reproduction():
    """The exact scenario a design review reproduced: 647 profitable
    days, 427 exact-zero abstention days, 7 genuine losses -- n=1081,
    matching S3's real profile. Confirms the fixed ES is NOT the
    old, diluted value.
    """
    rng = np.random.default_rng(0)
    profitable_losses = -np.abs(rng.normal(15, 8, 647))
    zero_losses = np.zeros(427)
    losing_losses = np.abs(rng.normal(10, 5, 7))
    all_losses = pd.Series(np.concatenate([profitable_losses, zero_losses, losing_losses]))

    result = rst.compute_empirical_var_es(all_losses, 0.95)
    # The old (buggy) tied-threshold method gave ES ~= 0.17 on this
    # exact data (confirmed before the fix was written); the correct
    # exact-m method gives ~1.38. Assert we're nowhere near the old,
    # diluted value.
    assert result["es"] > 1.0
    assert result["n_tail"] == pytest.approx(1081 * 0.05, rel=0.01)


def test_es_handles_fractional_m_correctly():
    """m = (1-c)*n is not generally an integer -- confirms the
    boundary observation gets correctly weighted by its fractional
    share, not silently rounded.
    """
    # n=10, c=0.9 -> m=1.0 exactly (no fractional boundary, sanity case)
    losses = pd.Series([50.0, 40.0, 30.0, 20.0, 10.0, 0.0, -10.0, -20.0, -30.0, -40.0])
    result = rst.compute_empirical_var_es(losses, 0.9)
    assert result["es"] == pytest.approx(50.0)  # worst 1 observation exactly

    # n=7, c=0.9 -> m=0.7 -- genuinely fractional, worst observation
    # gets 0.7 weight, nothing else contributes.
    losses2 = pd.Series([100.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    result2 = rst.compute_empirical_var_es(losses2, 0.9)
    assert result2["es"] == pytest.approx(100.0)  # only the single worst obs, weight 0.7, ES = sum/m = 70/0.7 = 100


# ---------------------------------------------------------------------
# validate_pnl_data -- fail-closed gate, same discipline as decomposition
# ---------------------------------------------------------------------
def _valid_pnl_df(n=10):
    from run_strategy_backtest import STRATEGIES
    rows = []
    for d in range(n):
        row = {"delivery_date": f"2023-06-{d+1:02d}"}
        for strat in STRATEGIES:
            row[f"{strat}_net_pnl"] = 0.0 if strat == "S0" else 10.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_validate_pnl_passes_on_clean_data():
    df = _valid_pnl_df()
    rst.validate_pnl_data(df)  # must not raise


def test_validate_pnl_catches_nan():
    df = _valid_pnl_df()
    df.loc[2, "S1_net_pnl"] = np.nan
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rst.validate_pnl_data(df)


def test_validate_pnl_catches_inf():
    df = _valid_pnl_df()
    df.loc[2, "S1_net_pnl"] = float("inf")
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rst.validate_pnl_data(df)


def test_validate_pnl_catches_duplicate_dates():
    df = _valid_pnl_df()
    df.loc[1, "delivery_date"] = df.loc[0, "delivery_date"]
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rst.validate_pnl_data(df)


def test_validate_pnl_catches_holdout_dates():
    df = _valid_pnl_df()
    df.loc[2, "delivery_date"] = "2026-01-01"
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rst.validate_pnl_data(df)


def test_validate_pnl_accepts_date_just_before_holdout():
    df = _valid_pnl_df()
    df.loc[2, "delivery_date"] = "2025-12-31"
    rst.validate_pnl_data(df)  # must not raise


def test_validate_pnl_catches_nonzero_s0():
    df = _valid_pnl_df()
    df.loc[2, "S0_net_pnl"] = 5.0
    with pytest.raises(ValueError, match="INPUT VALIDATION FAILED"):
        rst.validate_pnl_data(df)


# ---------------------------------------------------------------------
# Provenance import -- confirms reuse, not duplication, of the
# decomposition script's already-tested provenance logic
# ---------------------------------------------------------------------
def test_verify_source_provenance_is_imported_not_duplicated():
    src = ast.parse((REPO_ROOT / "run_strategy_tailrisk.py").read_text())
    imported_names = set()
    for node in ast.walk(src):
        if isinstance(node, ast.ImportFrom) and node.module == "run_strategy_decomposition":
            imported_names.update(alias.name for alias in node.names)
    assert "verify_source_provenance" in imported_names
