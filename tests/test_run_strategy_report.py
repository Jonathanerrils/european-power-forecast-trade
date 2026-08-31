"""Tests for run_strategy_report.py -- the read-only reporting layer.
Every compute function here is tested against hand-verifiable
synthetic data, and the architectural constraint (never calls any
decision-making function) is tested statically, the same way oracle
isolation is tested in tests/test_strategy_architecture.py.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_strategy_report as rsr


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Architectural constraint: never calls any decision-making function.
# This is exactly the bug class this project has repeatedly guarded
# against (oracle isolation) -- a static check, not just a docstring
# claim.
# ---------------------------------------------------------------------
def test_report_never_references_decision_making_functions():
    tree = ast.parse((REPO_ROOT / "run_strategy_report.py").read_text())
    forbidden = {"run_day", "best_point_pair", "best_uncertainty_pair", "realized_pnl", "oracle_pnl", "best_realized_pair"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            found.add(node.id)
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            found.add(node.attr)
    assert not found, f"run_strategy_report.py references decision-making function(s): {found}"


# ---------------------------------------------------------------------
# _fmt_pct / _fmt_eur -- the exact bug found and fixed: pandas silently
# converts None to NaN in a numeric column, and `x is not None` doesn't
# catch that, producing "nan%" instead of "N/A".
# ---------------------------------------------------------------------
def test_fmt_pct_handles_none():
    assert rsr._fmt_pct(None) == "N/A"


def test_fmt_pct_handles_nan_from_pandas_column_coercion():
    """The actual reproduced bug: None assigned in a dict, then
    coerced to NaN once placed in a pandas numeric column.
    """
    df = pd.DataFrame([{"x": None}, {"x": 0.847}])
    assert df["x"].dtype == np.float64  # confirms the coercion actually happened
    coerced_value = df.loc[0, "x"]
    assert rsr._fmt_pct(coerced_value) == "N/A"


def test_fmt_pct_formats_a_real_value():
    assert rsr._fmt_pct(0.847) == "84.7%"


def test_fmt_eur_handles_none_and_nan():
    assert rsr._fmt_eur(None) == "N/A"
    assert rsr._fmt_eur(float("nan")) == "N/A"


def test_fmt_eur_formats_a_real_value():
    assert rsr._fmt_eur(1234.5) == "EUR 1,234.50"


# ---------------------------------------------------------------------
# compute_primary_table
# ---------------------------------------------------------------------
def _synthetic_per_day():
    from run_strategy_backtest import STRATEGIES
    rows = []
    rng = np.random.default_rng(0)
    for d in range(20):
        row = {"delivery_date": f"2023-06-{d+1:02d}", "n_hours": 24, "oracle_pnl": 50.0}
        for strat in STRATEGIES:
            if strat == "S0":
                traded, net, gross = False, 0.0, 0.0
            else:
                traded = bool(rng.random() < 0.7)
                net = float(rng.normal(10, 30)) if traded else 0.0
                gross = net + 3.0 if traded else 0.0
            row[f"{strat}_traded"] = traded
            row[f"{strat}_net_pnl"] = net
            row[f"{strat}_gross_pnl"] = gross
        rows.append(row)
    return pd.DataFrame(rows)


def test_compute_primary_table_s0_never_trades():
    df = _synthetic_per_day()
    primary = rsr.compute_primary_table(df)
    s0 = primary[primary["strategy"] == "S0"].iloc[0]
    assert s0["trading_days"] == 0
    assert s0["no_trade_days"] == 20
    assert s0["net_pnl"] == 0.0
    assert pd.isna(s0["trade_hit_rate"])  # zero trades -> unset (None, coerced to NaN by pandas) -> N/A at display time, never 0%


def test_compute_primary_table_hand_verifiable():
    """A small, fully hand-verifiable case."""
    from run_strategy_backtest import STRATEGIES
    rows = [
        {"delivery_date": "d1", "n_hours": 24, "oracle_pnl": 10.0},
        {"delivery_date": "d2", "n_hours": 24, "oracle_pnl": 10.0},
        {"delivery_date": "d3", "n_hours": 24, "oracle_pnl": 10.0},
    ]
    # S1: trades on d1 (profit 20), d2 (loss -5), abstains d3
    vals = {"S1": [(True, 20.0), (True, -5.0), (False, 0.0)]}
    for strat in STRATEGIES:
        pairs = vals.get(strat, [(False, 0.0)] * 3)
        for row, (traded, net) in zip(rows, pairs):
            row[f"{strat}_traded"] = traded
            row[f"{strat}_net_pnl"] = net
            row[f"{strat}_gross_pnl"] = net + 1.0 if traded else 0.0
    df = pd.DataFrame(rows)
    primary = rsr.compute_primary_table(df)
    s1 = primary[primary["strategy"] == "S1"].iloc[0]
    assert s1["trading_days"] == 2
    assert s1["no_trade_days"] == 1
    assert s1["net_pnl"] == pytest.approx(15.0)
    assert s1["profitable_day_rate"] == pytest.approx(1 / 3)  # 1 profitable day out of 3 total
    assert s1["trade_hit_rate"] == pytest.approx(0.5)  # 1 of 2 TRADES was profitable
    assert s1["worst_day"] == pytest.approx(-5.0)


# ---------------------------------------------------------------------
# compute_named_deltas
# ---------------------------------------------------------------------
def test_compute_named_deltas():
    from run_strategy_backtest import STRATEGIES
    df = pd.DataFrame([{f"{s}_net_pnl": v for s, v in zip(STRATEGIES, [0, 10, 15, 5, 8, 2])}])
    deltas = rsr.compute_named_deltas(df)
    assert deltas["delta_forecast"] == pytest.approx(5.0)   # S2-S1 = 15-10
    assert deltas["delta_uncertainty"] == pytest.approx(-10.0)  # S3-S2 = 5-15
    assert deltas["delta_tier1"] == pytest.approx(-7.0)  # S4-S2 = 8-15
    assert deltas["delta_tier1_u"] == pytest.approx(-3.0)  # S5-S3 = 2-5


# ---------------------------------------------------------------------
# compute_drawdown
# ---------------------------------------------------------------------
def test_compute_drawdown_hand_verifiable():
    """Daily P&L: [10, -5, 20, -30, 5]. Cumulative: [10, 5, 25, -5, 0].
    Running max: [10, 10, 25, 25, 25]. Drawdown: [0, -5, 0, -30, -25].
    Max drawdown = -30 (at day 4, from peak of 25 to -5).
    """
    series = pd.Series([10.0, -5.0, 20.0, -30.0, 5.0])
    dd = rsr.compute_drawdown(series)
    assert dd["worst_day"] == pytest.approx(-30.0)
    assert dd["max_drawdown"] == pytest.approx(-30.0)
    assert dd["worst_rolling_5day"] == pytest.approx(0.0)  # only one full 5-day window: sum = 0


def test_compute_drawdown_too_short_for_rolling_5day():
    series = pd.Series([10.0, -5.0])
    dd = rsr.compute_drawdown(series)
    assert dd["worst_rolling_5day"] is None


# ---------------------------------------------------------------------
# compute_concentration
# ---------------------------------------------------------------------
def test_compute_concentration_positive_total():
    series = pd.Series([100.0, 5.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # total=116, top1=100
    conc = rsr.compute_concentration(series)
    assert conc["top_1pct_abs_eur"] == pytest.approx(100.0)
    assert conc["top_1pct_pct_of_total"] == pytest.approx(100.0 / 116.0)


def test_compute_concentration_nonpositive_total_reports_na_percentage():
    """Total <= 0: percentage is N/A (None), but absolute EUR
    contribution is still reported -- per docs/economic_contract_v1.md.
    """
    series = pd.Series([-50.0, 10.0, 5.0, -20.0])  # total = -55
    conc = rsr.compute_concentration(series)
    assert conc["top_1pct_pct_of_total"] is None
    assert conc["top_1pct_abs_eur"] is not None  # absolute value still computed


# ---------------------------------------------------------------------
# compute_oracle_value_capture
# ---------------------------------------------------------------------
def test_compute_oracle_value_capture():
    from run_strategy_backtest import STRATEGIES
    df = pd.DataFrame([{
        "oracle_pnl": 20.0,
        **{f"{s}_net_pnl": v for s, v in zip(STRATEGIES, [0, 15, 18, 10, 12, 5])},
    }])
    vcr = rsr.compute_oracle_value_capture(df)
    assert vcr["oracle_total"] == pytest.approx(20.0)
    assert vcr["S1_vcr"] == pytest.approx(15.0 / 20.0)
    assert vcr["S0_vcr"] == pytest.approx(0.0)


def test_compute_oracle_value_capture_zero_oracle_returns_none():
    from run_strategy_backtest import STRATEGIES
    df = pd.DataFrame([{"oracle_pnl": 0.0, **{f"{s}_net_pnl": 0.0 for s in STRATEGIES}}])
    vcr = rsr.compute_oracle_value_capture(df)
    assert vcr["S1_vcr"] is None


# ---------------------------------------------------------------------
# compute_split
# ---------------------------------------------------------------------
def test_compute_split_partitions_correctly():
    from run_strategy_backtest import STRATEGIES
    rows = []
    for d in range(4):
        row = {"is_flag": d < 2}
        for s, v in zip(STRATEGIES, [0, 10, 15, 5, 8, 2]):
            row[f"{s}_net_pnl"] = v * (d + 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    split = rsr.compute_split(df, "is_flag")
    assert split["true"]["n_days"] == 2
    assert split["false"]["n_days"] == 2


def test_compute_split_handles_empty_subset():
    from run_strategy_backtest import STRATEGIES
    df = pd.DataFrame([{"is_flag": True, **{f"{s}_net_pnl": 1.0 for s in STRATEGIES}}])
    split = rsr.compute_split(df, "is_flag")
    assert split["false"]["n_days"] == 0


# ---------------------------------------------------------------------
# resolve_run_args / load_all_artifacts
# ---------------------------------------------------------------------
def test_resolve_run_args_requires_exactly_four():
    with pytest.raises(SystemExit):
        rsr.resolve_run_args(["a", "b", "c"])
    with pytest.raises(SystemExit):
        rsr.resolve_run_args(["a", "b", "c", "d", "e"])


def test_load_all_artifacts_raises_clearly_when_backtest_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rsr, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing"):
        rsr.load_all_artifacts("delu_features", "never_run", "sens", "decomp")
