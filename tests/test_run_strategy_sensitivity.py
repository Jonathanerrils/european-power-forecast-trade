"""Tests for run_strategy_sensitivity.py -- the base-case reproduction
gate is the most important thing here, so it's tested both ways: that
a genuine match passes, and that a real mismatch is caught, not just
asserted to work.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_strategy_sensitivity as rss


def test_resolve_run_args_requires_exactly_five():
    with pytest.raises(SystemExit):
        rss.resolve_run_args(["a", "b", "c", "d"])
    with pytest.raises(SystemExit):
        rss.resolve_run_args(["a", "b", "c", "d", "e", "f"])


def test_resolve_run_args_valid_call():
    result = rss.resolve_run_args(["xg", "unc", "unc_t1", "base_out", "out"])
    assert result == ("xg", "unc", "unc_t1", "base_out", "out")


def test_grid_has_exactly_nine_combinations():
    from itertools import product
    grid = list(product(rss.ETA_RT_GRID, rss.C_GRID))
    assert len(grid) == 9
    assert rss.BASE_CASE in grid


def test_compute_deltas():
    df = pd.DataFrame({
        "S1_net_pnl": [10.0, 20.0], "S2_net_pnl": [15.0, 25.0],
        "S3_net_pnl": [5.0, 10.0], "S4_net_pnl": [8.0, 12.0], "S5_net_pnl": [2.0, 3.0],
    })
    deltas = rss.compute_deltas(df)
    assert deltas["delta_forecast"] == pytest.approx((15+25) - (10+20))
    assert deltas["delta_uncertainty"] == pytest.approx((5+10) - (15+25))
    assert deltas["delta_tier1"] == pytest.approx((8+12) - (15+25))
    assert deltas["delta_tier1_u"] == pytest.approx((2+3) - (5+10))


# ---------------------------------------------------------------------
# verify_base_case_reproduction -- the actual gate, tested both ways
# ---------------------------------------------------------------------
def _make_results_df(net_pnls: dict, ij: dict = None) -> pd.DataFrame:
    row = {"delivery_date": "2023-06-01", "n_hours": 24, "oracle_pnl": 100.0}
    ij = ij or {}
    for strat in rss.STRATEGIES:
        row[f"{strat}_traded"] = True
        row[f"{strat}_i"] = ij.get(strat, (2, 5))[0]
        row[f"{strat}_j"] = ij.get(strat, (2, 5))[1]
        row[f"{strat}_net_pnl"] = net_pnls.get(strat, 0.0)
        row[f"{strat}_gross_pnl"] = net_pnls.get(strat, 0.0) + 5.0
    return pd.DataFrame([row])


def test_reproduction_check_passes_on_genuine_match(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "REPO_ROOT", tmp_path)
    saved_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "base_run"
    saved_dir.mkdir(parents=True)
    net_pnls = {"S0": 0.0, "S1": 10.0, "S2": 15.0, "S3": 5.0, "S4": 8.0, "S5": 2.0}
    saved = _make_results_df(net_pnls)
    saved.to_csv(saved_dir / "per_day_results.csv", index=False)

    computed = _make_results_df(net_pnls)  # identical
    rss.verify_base_case_reproduction(computed, "base_run", "delu_features")  # must not raise


def test_reproduction_check_catches_a_real_mismatch(tmp_path, monkeypatch):
    """The gate must actually fire on a genuine discrepancy, not just
    pass trivially on matching inputs -- constructs a case where ONE
    strategy's net_pnl differs by a small but real amount.
    """
    monkeypatch.setattr(rss, "REPO_ROOT", tmp_path)
    saved_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "base_run"
    saved_dir.mkdir(parents=True)
    saved_pnls = {"S0": 0.0, "S1": 10.0, "S2": 15.0, "S3": 5.0, "S4": 8.0, "S5": 2.0}
    saved = _make_results_df(saved_pnls)
    saved.to_csv(saved_dir / "per_day_results.csv", index=False)

    computed_pnls = dict(saved_pnls)
    computed_pnls["S2"] = 15.5  # deliberately different
    computed = _make_results_df(computed_pnls)

    with pytest.raises(AssertionError, match="BASE-CASE REPRODUCTION FAILED"):
        rss.verify_base_case_reproduction(computed, "base_run", "delu_features")


def test_reproduction_check_catches_a_traded_flag_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "REPO_ROOT", tmp_path)
    saved_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "base_run"
    saved_dir.mkdir(parents=True)
    saved = _make_results_df({"S0": 0.0, "S1": 10.0, "S2": 15.0, "S3": 5.0, "S4": 8.0, "S5": 2.0})
    saved.to_csv(saved_dir / "per_day_results.csv", index=False)

    computed = saved.copy()
    computed.loc[0, "S3_traded"] = False  # deliberately flip a decision

    with pytest.raises(AssertionError, match="BASE-CASE REPRODUCTION FAILED"):
        rss.verify_base_case_reproduction(computed, "base_run", "delu_features")


def test_reproduction_check_raises_clearly_when_base_not_yet_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "REPO_ROOT", tmp_path)
    computed = _make_results_df({"S1": 10.0})
    with pytest.raises(FileNotFoundError, match="No saved base results"):
        rss.verify_base_case_reproduction(computed, "never_run", "delu_features")


def test_reproduction_check_catches_mismatched_day_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "REPO_ROOT", tmp_path)
    saved_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "base_run"
    saved_dir.mkdir(parents=True)
    saved = _make_results_df({"S1": 10.0})
    saved.to_csv(saved_dir / "per_day_results.csv", index=False)

    computed = _make_results_df({"S1": 10.0})
    computed["delivery_date"] = "2023-06-02"  # a completely different day

    with pytest.raises(AssertionError, match="common-day sets don't even match"):
        rss.verify_base_case_reproduction(computed, "base_run", "delu_features")


# ---------------------------------------------------------------------
# Strengthened checks: (i, j), gross P&L, oracle, n_hours, four deltas --
# the exact gaps a design review found in the earlier, weaker version
# (which only checked traded flags and net P&L).
# ---------------------------------------------------------------------
def test_reproduction_check_catches_different_pair_with_identical_net_pnl():
    """THE specific gap the earlier version had: two different (i, j)
    choices can coincidentally produce the same net P&L (e.g. on a day
    with more than one equally-profitable cycle) -- net P&L alone
    would have missed this entirely.
    """
    def make(tmp_path):
        saved_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "base_run"
        saved_dir.mkdir(parents=True)
        saved = _make_results_df({"S2": 20.0}, ij={"S2": (2, 5)})
        saved.to_csv(saved_dir / "per_day_results.csv", index=False)
        computed = _make_results_df({"S2": 20.0}, ij={"S2": (3, 6)})  # different pair, SAME net P&L
        return computed

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        computed = make(tmp_path)
        import run_strategy_sensitivity as rss_module
        rss_module.REPO_ROOT = tmp_path
        with pytest.raises(AssertionError, match="BASE-CASE REPRODUCTION FAILED"):
            rss_module.verify_base_case_reproduction(computed, "base_run", "delu_features")


def test_reproduction_check_catches_oracle_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "REPO_ROOT", tmp_path)
    saved_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "base_run"
    saved_dir.mkdir(parents=True)
    saved = _make_results_df({"S1": 10.0})
    saved.to_csv(saved_dir / "per_day_results.csv", index=False)

    computed = _make_results_df({"S1": 10.0})
    computed["oracle_pnl"] = 999.0  # deliberately different oracle value

    with pytest.raises(AssertionError, match="BASE-CASE REPRODUCTION FAILED"):
        rss.verify_base_case_reproduction(computed, "base_run", "delu_features")


def test_reproduction_check_catches_gross_pnl_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "REPO_ROOT", tmp_path)
    saved_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "base_run"
    saved_dir.mkdir(parents=True)
    saved = _make_results_df({"S2": 20.0})
    saved.to_csv(saved_dir / "per_day_results.csv", index=False)

    computed = saved.copy()
    computed.loc[0, "S2_gross_pnl"] = computed.loc[0, "S2_gross_pnl"] + 1.0

    with pytest.raises(AssertionError, match="BASE-CASE REPRODUCTION FAILED"):
        rss.verify_base_case_reproduction(computed, "base_run", "delu_features")


def test_aggregate_deltas_are_determined_by_checked_per_day_results():
    """NOT a failure-mode test -- this documents WHY a separate
    delta-only mismatch case can't be meaningfully constructed: the
    four deltas are pure sums of per-day columns already checked above
    (net_pnl for each strategy). If every per-day value matches, the
    deltas are mathematically forced to match too -- there is no way
    for compute_deltas() to diverge from saved vs. computed without a
    per-day value already having diverged first, which the checks
    above already catch independently.
    """
    def make():
        row_saved = {"delivery_date": "2023-06-01", "n_hours": 24, "oracle_pnl": 100.0}
        row_computed = {"delivery_date": "2023-06-01", "n_hours": 24, "oracle_pnl": 100.0}
        for strat, pnl_saved, pnl_computed in [
            ("S0", 0.0, 0.0), ("S1", 10.0, 10.0), ("S2", 20.0, 20.0),
            ("S3", 5.0, 5.0), ("S4", 8.0, 8.0), ("S5", 2.0, 2.0),
        ]:
            for row, pnl in ((row_saved, pnl_saved), (row_computed, pnl_computed)):
                row[f"{strat}_traded"] = True
                row[f"{strat}_i"] = 2
                row[f"{strat}_j"] = 5
                row[f"{strat}_net_pnl"] = pnl
                row[f"{strat}_gross_pnl"] = pnl + 5.0
        return pd.DataFrame([row_saved]), pd.DataFrame([row_computed])

    saved, computed = make()
    assert (saved["S1_net_pnl"] == computed["S1_net_pnl"]).all()  # per-day values genuinely identical
    # This case can't actually diverge on deltas without diverging on a
    # per-day value first (deltas are pure sums of already-checked
    # columns) -- confirming that structurally, rather than asserting
    # an artificial delta-only mismatch that couldn't arise in practice.
    saved_deltas = rss.compute_deltas(saved)
    computed_deltas = rss.compute_deltas(computed)
    assert saved_deltas == computed_deltas


def test_reproduction_check_raises_clearly_when_saved_file_predates_ij_fix(tmp_path, monkeypatch):
    """A base run saved BEFORE the (i, j)-saving fix (missing those
    columns entirely) must fail with a clear, actionable message --
    not a confusing KeyError deep inside pandas.
    """
    monkeypatch.setattr(rss, "REPO_ROOT", tmp_path)
    saved_dir = tmp_path / "outputs" / "strategy" / "delu_features" / "old_base_run"
    saved_dir.mkdir(parents=True)
    old_format = pd.DataFrame([{
        "delivery_date": "2023-06-01", "n_hours": 24, "oracle_pnl": 100.0,
        "S1_traded": True, "S1_net_pnl": 10.0, "S1_gross_pnl": 15.0,
    }])
    old_format.to_csv(saved_dir / "per_day_results.csv", index=False)

    computed = _make_results_df({"S1": 10.0})
    with pytest.raises(ValueError, match="predates the .i, j.-saving fix"):
        rss.verify_base_case_reproduction(computed, "old_base_run", "delu_features")
