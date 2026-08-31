"""Run locally: python run_strategy_report.py <backtest_run_version> <sensitivity_run_version> <decomposition_run_version> <output_run_version>

Example:
  python run_strategy_report.py strategy_backtest_v2 strategy_sensitivity_v1 strategy_decomposition_v1 strategy_report_v1

READ-ONLY reporting layer. Consumes ONLY already-saved artifacts:
per_day_results.csv (strategy_backtest_v2), sensitivity_grid_summary.csv
(strategy_sensitivity_v1), decomposition_results.json
(strategy_decomposition_v1), and the raw hourly price series (via
run_strategy_backtest.load_combined_data, reused for READING only) to
identify regime/stress days -- delivery_date alone is enough for the
pre/post-2025-10-01 split, but negative-price/>200/>500 stress days
need the actual hourly prices, which per_day_results.csv does not
persist.

This script NEVER calls run_day(), best_point_pair(), or any other
decision-making function -- it only aggregates and summarizes numbers
that already exist on disk. xgboost_run_version/uncertainty_run_version/
tier1_uncertainty_run_version are read automatically from
strategy_backtest_v2's own saved manifest rather than re-specified,
both for convenience and as a consistency check that the report is
summarizing the run it claims to be.

Report sections, in order: (1) provenance/population, (2) primary
strategy table, (3) four pre-registered comparisons, (4) 3x3
sensitivity (all four delta tables), (5) uncertainty decomposition
(S2<->S3, S4<->S5 -- reported with the exact language established:
S5-S4 is a DIFFERENT quantity from the pre-registered delta_tier1_u=S5-S3,
never substituted for it), (6) downside/drawdown, (7) oracle/value
capture, (8) concentration, (9) regime split, (10) stress-day split,
(11) explicit limitations.

Deliberately NOT included: Sharpe ratio, annualized return, ROI, or
any capital-efficiency metric -- there is still no capital/exposure
model to support them (docs/economic_contract_v1.md).

STANDING CAVEAT: reads only already-saved development-period results;
never touches 2026.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from run_strategy_backtest import STRATEGIES, load_combined_data, build_common_evaluation_days
from src.models import TARGET_COL
from src.utils import load_config, setup_logging, REPO_ROOT

CUTOVER_LOCAL_DATE = "2025-10-01"
NAMED_DELTAS = [
    ("delta_forecast", "S2", "S1", "XGBoost value: S2 - S1 (PRIMARY)"),
    ("delta_uncertainty", "S3", "S2", "Uncertainty value: S3 - S2"),
    ("delta_tier1", "S4", "S2", "Tier-1 cost: S4 - S2"),
    ("delta_tier1_u", "S5", "S3", "Tier-1 uncertainty cost: S5 - S3"),
]


def resolve_run_args(args: list) -> tuple:
    if len(args) != 4:
        raise SystemExit(
            "Usage:\n"
            "  python run_strategy_report.py <backtest_run_version> <sensitivity_run_version> "
            "<decomposition_run_version> <output_run_version>\n\n"
            "Example:\n"
            "  python run_strategy_report.py strategy_backtest_v2 strategy_sensitivity_v1 "
            "strategy_decomposition_v1 strategy_report_v1"
        )
    return tuple(args)


def load_all_artifacts(input_stem: str, backtest_run_version: str, sensitivity_run_version: str, decomposition_run_version: str) -> dict:
    """Loads every already-saved artifact this report needs. Raises
    clearly if any is missing -- never regenerates a missing one.
    """
    backtest_dir = REPO_ROOT / "outputs" / "strategy" / input_stem / backtest_run_version
    per_day_path = backtest_dir / "per_day_results.csv"
    manifest_path = backtest_dir / "strategy_backtest_manifest.json"
    if not per_day_path.exists():
        raise FileNotFoundError(f"Missing {per_day_path}. Run run_strategy_backtest.py first.")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}.")
    per_day = pd.read_csv(per_day_path)
    with open(manifest_path) as f:
        backtest_manifest = json.load(f)

    sensitivity_path = REPO_ROOT / "outputs" / "strategy" / input_stem / sensitivity_run_version / "sensitivity_grid_summary.csv"
    if not sensitivity_path.exists():
        raise FileNotFoundError(f"Missing {sensitivity_path}. Run run_strategy_sensitivity.py first.")
    sensitivity_grid = pd.read_csv(sensitivity_path)

    decomposition_path = REPO_ROOT / "outputs" / "strategy" / input_stem / decomposition_run_version / "decomposition_results.json"
    if not decomposition_path.exists():
        raise FileNotFoundError(f"Missing {decomposition_path}. Run run_strategy_decomposition.py first.")
    with open(decomposition_path) as f:
        decomposition = json.load(f)

    return {
        "per_day": per_day,
        "backtest_manifest": backtest_manifest,
        "sensitivity_grid": sensitivity_grid,
        "decomposition": decomposition,
    }


def attach_regime_and_stress_flags(per_day: pd.DataFrame, input_stem: str, backtest_manifest: dict) -> pd.DataFrame:
    """Adds is_post_cutover, is_negative_price_day, is_gt200_day,
    is_gt500_day columns -- all defined by realized MARKET REGIME
    (independent of which interval, if any, the strategy actually
    selected), per docs/economic_contract_v1.md's explicit "negative-
    price day" definition. Requires the raw hourly price series, which
    per_day_results.csv does not persist -- read via
    load_combined_data (READ-ONLY reuse; no decision-making function
    is called here).
    """
    combined = load_combined_data(
        input_stem,
        backtest_manifest["xgboost_run_version"],
        backtest_manifest["uncertainty_run_version"],
        backtest_manifest["tier1_uncertainty_run_version"],
    )
    day_info = build_common_evaluation_days(combined)
    hourly = day_info["df_with_local_time"]

    stress_by_day = hourly.groupby("delivery_date")[TARGET_COL].agg(
        is_negative_price_day=lambda s: bool((s < 0).any()),
        is_gt200_day=lambda s: bool((s > 200).any()),
        is_gt500_day=lambda s: bool((s > 500).any()),
    ).reset_index()
    stress_by_day["delivery_date"] = stress_by_day["delivery_date"].astype(str)

    per_day = per_day.copy()
    per_day["delivery_date"] = per_day["delivery_date"].astype(str)
    per_day = per_day.merge(stress_by_day, on="delivery_date", how="left")
    per_day["is_post_cutover"] = per_day["delivery_date"] >= CUTOVER_LOCAL_DATE
    return per_day


def compute_primary_table(per_day: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_common_days = len(per_day)
    for strat in STRATEGIES:
        net = per_day[f"{strat}_net_pnl"]
        gross = per_day[f"{strat}_gross_pnl"]
        traded = per_day[f"{strat}_traded"].astype(bool)
        n_traded = int(traded.sum())
        n_no_trade = n_common_days - n_traded
        profitable_days = int((net > 0).sum())
        trade_hit_rate = (net[traded] > 0).mean() if n_traded > 0 else None  # N/A on zero trades, not 0%
        rows.append({
            "strategy": strat,
            "trading_days": n_traded,
            "no_trade_days": n_no_trade,
            "gross_pnl": float(gross.sum()),
            "net_pnl": float(net.sum()),
            "mean_daily_pnl": float(net.mean()),
            "median_daily_pnl": float(net.median()),
            "profitable_day_rate": profitable_days / n_common_days,
            "trade_hit_rate": trade_hit_rate,
            "worst_day": float(net.min()),
        })
    return pd.DataFrame(rows)


def compute_named_deltas(per_day: pd.DataFrame) -> dict:
    return {
        name: float(per_day[f"{a}_net_pnl"].sum() - per_day[f"{b}_net_pnl"].sum())
        for name, a, b, _ in NAMED_DELTAS
    }


def compute_drawdown(net_pnl_series: pd.Series) -> dict:
    """Worst single day, worst rolling 5-day sum, and maximum drawdown
    on cumulative P&L (peak-to-trough, not peak-to-current).
    """
    cumulative = net_pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    rolling_5day = net_pnl_series.rolling(5).sum()
    return {
        "worst_day": float(net_pnl_series.min()),
        "worst_rolling_5day": float(rolling_5day.min()) if len(net_pnl_series) >= 5 else None,
        "max_drawdown": float(drawdown.min()),
    }


def compute_concentration(net_pnl_series: pd.Series) -> dict:
    """% of total P&L from best 1%/5%/10% of days -- reported only
    when total net P&L > 0 (a percentage-of-total is meaningless or
    misleading otherwise); N/A plus absolute EUR contribution
    otherwise, per docs/economic_contract_v1.md.
    """
    total = net_pnl_series.sum()
    sorted_desc = net_pnl_series.sort_values(ascending=False)
    n = len(sorted_desc)
    result = {}
    for pct, label in [(0.01, "top_1pct"), (0.05, "top_5pct"), (0.10, "top_10pct")]:
        k = max(1, int(np.ceil(n * pct)))
        top_sum = float(sorted_desc.iloc[:k].sum())
        result[f"{label}_abs_eur"] = top_sum
        result[f"{label}_pct_of_total"] = (top_sum / total) if total > 0 else None
    return result


def compute_oracle_value_capture(per_day: pd.DataFrame) -> dict:
    oracle_sum = float(per_day["oracle_pnl"].sum())
    result = {"oracle_total": oracle_sum}
    for strat in STRATEGIES:
        strat_sum = float(per_day[f"{strat}_net_pnl"].sum())
        result[f"{strat}_vcr"] = (strat_sum / oracle_sum) if oracle_sum > 0 else None
    return result


def compute_split(per_day: pd.DataFrame, mask_col: str) -> dict:
    """Named deltas + primary-table-style summary, computed separately
    for the subset of days where mask_col is True vs False.
    """
    result = {}
    for label, subset in [("true", per_day[per_day[mask_col]]), ("false", per_day[~per_day[mask_col]])]:
        if len(subset) == 0:
            result[label] = {"n_days": 0}
            continue
        deltas = compute_named_deltas(subset)
        totals = {f"{s}_net_pnl": float(subset[f"{s}_net_pnl"].sum()) for s in STRATEGIES}
        result[label] = {"n_days": len(subset), **totals, **deltas}
    return result


def _fmt_pct(x):
    return f"{x*100:.1f}%" if pd.notna(x) else "N/A"


def _fmt_eur(x):
    return f"EUR {x:,.2f}" if pd.notna(x) else "N/A"


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    backtest_run_version, sensitivity_run_version, decomposition_run_version, output_run_version = resolve_run_args(sys.argv[1:])
    input_stem = "delu_features"

    out_dir = REPO_ROOT / "outputs" / "strategy" / input_stem / output_run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already contains results. Pass a new output_run_version.")

    logger.info("Loading all already-saved artifacts (read-only)...")
    artifacts = load_all_artifacts(input_stem, backtest_run_version, sensitivity_run_version, decomposition_run_version)
    per_day = attach_regime_and_stress_flags(artifacts["per_day"], input_stem, artifacts["backtest_manifest"])
    n_days = len(per_day)

    report_lines = []
    def out(line=""):
        print(line)
        report_lines.append(line)

    out("=" * 78)
    out(f"STRATEGY REPORT: {output_run_version}")
    out(f"Source: {backtest_run_version} / {sensitivity_run_version} / {decomposition_run_version}")
    out("=" * 78)

    # 1. Provenance and evaluation population
    out("\n--- 1. PROVENANCE AND EVALUATION POPULATION ---")
    out(f"Common development evaluation days: {n_days}")
    out(f"eta_rt={artifacts['backtest_manifest']['eta_rt']}, c=EUR{artifacts['backtest_manifest']['c']}")
    out(f"Holdout used: {artifacts['backtest_manifest']['holdout_used']} (must be False)")
    out(f"Structural invariants: {artifacts['backtest_manifest']['structural_invariants']}")

    # 2. Primary strategy table
    out("\n--- 2. PRIMARY STRATEGY TABLE ---")
    primary = compute_primary_table(per_day)
    display_primary = primary.copy()
    display_primary["profitable_day_rate"] = display_primary["profitable_day_rate"].map(_fmt_pct)
    display_primary["trade_hit_rate"] = display_primary["trade_hit_rate"].map(_fmt_pct)
    for col in ("gross_pnl", "net_pnl", "mean_daily_pnl", "median_daily_pnl", "worst_day"):
        display_primary[col] = display_primary[col].map(lambda x: f"{x:,.2f}")
    out(display_primary.to_string(index=False))

    # 3. Four pre-registered comparisons
    out("\n--- 3. FOUR PRE-REGISTERED COMPARISONS ---")
    deltas = compute_named_deltas(per_day)
    for name, _, _, label in NAMED_DELTAS:
        out(f"  {label}: {_fmt_eur(deltas[name])}")

    # 4. 3x3 sensitivity
    out("\n--- 4. 3x3 SENSITIVITY GRID (all four deltas) ---")
    grid = artifacts["sensitivity_grid"]
    out(grid[["eta_rt", "c", "delta_forecast", "delta_uncertainty", "delta_tier1", "delta_tier1_u"]].to_string(index=False))
    for delta_name in ("delta_forecast", "delta_uncertainty", "delta_tier1", "delta_tier1_u"):
        n_pos = int((grid[delta_name] > 0).sum())
        n_neg = int((grid[delta_name] < 0).sum())
        out(f"  {delta_name}: +{n_pos}/9  -{n_neg}/9")

    # 5. Uncertainty decomposition
    out("\n--- 5. UNCERTAINTY DECOMPOSITION ---")
    for key, r in artifacts["decomposition"].items():
        out(f"  {r['point_strategy']} <-> {r['uncertainty_strategy']}:")
        out(f"    Profits forgone: {_fmt_eur(r['profits_forgone'])}  Losses avoided: {_fmt_eur(r['losses_avoided'])}  "
            f"Pair-selection: {_fmt_eur(r['pair_selection_effect'])}")
        out(f"    Reconciled gap ({r['uncertainty_strategy']}-{r['point_strategy']}): {_fmt_eur(r['total_gap'])}")
    out("  NOTE: the S4<->S5 gap above (S5-S4) is a DIFFERENT quantity from the pre-registered")
    out("  delta_tier1_u (S5-S3) reported in section 3 -- never substitute one for the other.")

    # 6. Downside/drawdown
    out("\n--- 6. DOWNSIDE / DRAWDOWN ---")
    for strat in STRATEGIES:
        dd = compute_drawdown(per_day[f"{strat}_net_pnl"])
        out(f"  {strat}: worst_day={_fmt_eur(dd['worst_day'])}  worst_5day={_fmt_eur(dd['worst_rolling_5day'])}  "
            f"max_drawdown={_fmt_eur(dd['max_drawdown'])}")

    # 7. Oracle / value capture
    out("\n--- 7. ORACLE / VALUE CAPTURE ---")
    vcr = compute_oracle_value_capture(per_day)
    out(f"  Oracle total (ex-post diagnostic, not an executable strategy): {_fmt_eur(vcr['oracle_total'])}")
    for strat in STRATEGIES:
        out(f"  {strat} value capture ratio: {vcr[f'{strat}_vcr']:.3f}" if vcr[f"{strat}_vcr"] is not None else f"  {strat} value capture ratio: N/A")

    # 8. Concentration
    out("\n--- 8. CONCENTRATION ---")
    for strat in STRATEGIES:
        conc = compute_concentration(per_day[f"{strat}_net_pnl"])
        out(f"  {strat}: top1%={_fmt_eur(conc['top_1pct_abs_eur'])} ({_fmt_pct(conc['top_1pct_pct_of_total'])})  "
            f"top5%={_fmt_eur(conc['top_5pct_abs_eur'])} ({_fmt_pct(conc['top_5pct_pct_of_total'])})  "
            f"top10%={_fmt_eur(conc['top_10pct_abs_eur'])} ({_fmt_pct(conc['top_10pct_pct_of_total'])})")

    # 9. Regime split
    out("\n--- 9. REGIME SPLIT (pre/post 2025-10-01) ---")
    regime = compute_split(per_day, "is_post_cutover")
    out(f"  Pre-cutover ({regime['false']['n_days']} days): " +
        ", ".join(f"{name}={_fmt_eur(regime['false'].get(name))}" for name, _, _, _ in NAMED_DELTAS))
    out(f"  Post-cutover ({regime['true']['n_days']} days): " +
        ", ".join(f"{name}={_fmt_eur(regime['true'].get(name))}" for name, _, _, _ in NAMED_DELTAS))

    # 10. Stress-day split
    out("\n--- 10. STRESS-DAY SPLIT ---")
    for mask_col, label in [("is_negative_price_day", "Negative-price days"), ("is_gt200_day", ">EUR200 days"), ("is_gt500_day", ">EUR500 days")]:
        split = compute_split(per_day, mask_col)
        out(f"  {label}: {split['true']['n_days']} stress day(s), delta_forecast={_fmt_eur(split['true'].get('delta_forecast'))} "
            f"vs. {split['false']['n_days']} normal day(s), delta_forecast={_fmt_eur(split['false'].get('delta_forecast'))}")

    # 11. Limitations
    out("\n--- 11. EXPLICIT LIMITATIONS ---")
    out("  - Price-taking assumption: no market impact modeled.")
    out("  - Market-access costs (exchange/clearing/brokerage) excluded (C_market=0).")
    out("  - Stylised single-cycle battery: one charge, one discharge, per delivery day.")
    out("  - Development sample only (2023-2025) -- 2026 holdout untouched, not a substitute for it.")
    out("  - Tier-2 (wind/solar-derived) feature point-in-time availability remains unresolved --")
    out("    a real, load-bearing limitation on the Full model's edge, not a formality.")
    out("  - No capital/exposure model: Sharpe, annualized return, and ROI are deliberately excluded.")

    out("\n" + "=" * 78)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "strategy_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    primary.to_csv(out_dir / "primary_table.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backtest_run_version": backtest_run_version,
        "sensitivity_run_version": sensitivity_run_version,
        "decomposition_run_version": decomposition_run_version,
        "output_run_version": output_run_version,
        "read_only": True,
        "holdout_used": False,
    }
    with open(out_dir / "strategy_report_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved full report + manifest to {out_dir}")


if __name__ == "__main__":
    main()
