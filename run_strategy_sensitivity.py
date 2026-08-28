"""Run locally: python run_strategy_sensitivity.py <xgboost_run_version> <uncertainty_run_version> <tier1_uncertainty_run_version> <base_output_run_version> <output_run_version>

Example:
  python run_strategy_sensitivity.py xgboost_v1_a03fix uncertainty_selected_v1 uncertainty_tier1_robustness_v2 strategy_backtest_v1 strategy_sensitivity_v1

PRE-REGISTERED 3x3 sensitivity grid (docs/economic_contract_v1.md):
eta_rt in {0.70, 0.85, 0.92}, c in {5, 10, 15} -- all NINE combinations,
computed once, no cherry-picking, no rule changes. Reuses
run_strategy_backtest.py's exact functions (load_combined_data,
build_common_evaluation_days, run_backtest_for_day,
verify_structural_invariants) rather than duplicating the economic
logic -- there is exactly one implementation of "how a day's strategy
decision and P&L are computed" in this project, used by both scripts.

CRITICAL GATE, run before anything else: the (eta_rt=0.85, c=10) cell
--the ALREADY-COMPUTED primary case-- must reproduce
<base_output_run_version>'s saved results EXACTLY (same per-day
decisions, same P&L, same four deltas). If it doesn't, STOP before
interpreting any of the nine cells -- this is the strategy-layer
equivalent of run_xgboost.py's baseline_v1 reproduction check.

Also asserts, for EVERY one of the nine cells: identical common
evaluation days to the base run, no 2026 timestamps, i<j on every
trade, S0 always 0, oracle dominance holds for every strategy/day, and
that exactly nine combinations were computed (no missing/extra).

STANDING CAVEAT: never touches 2026.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from run_strategy_backtest import (
    STRATEGIES,
    build_common_evaluation_days,
    load_combined_data,
    run_backtest_for_day,
    verify_structural_invariants,
)
from src.strategy import assert_no_holdout_access
from src.utils import load_config, setup_logging, REPO_ROOT

# PRE-REGISTERED, fixed before this script has ever been run against
# real data -- do not add, remove, or reorder after seeing a result.
ETA_RT_GRID = [0.70, 0.85, 0.92]
C_GRID = [5, 10, 15]
BASE_CASE = (0.85, 10)


def resolve_run_args(args: list) -> tuple:
    if len(args) != 5:
        raise SystemExit(
            "Usage:\n"
            "  python run_strategy_sensitivity.py <xgboost_run_version> <uncertainty_run_version> "
            "<tier1_uncertainty_run_version> <base_output_run_version> <output_run_version>\n\n"
            "Example:\n"
            "  python run_strategy_sensitivity.py xgboost_v1_a03fix uncertainty_selected_v1 "
            "uncertainty_tier1_robustness_v2 strategy_backtest_v1 strategy_sensitivity_v1"
        )
    return tuple(args)


def compute_deltas(results_df: pd.DataFrame) -> dict:
    return {
        "delta_forecast": float(results_df["S2_net_pnl"].sum() - results_df["S1_net_pnl"].sum()),
        "delta_uncertainty": float(results_df["S3_net_pnl"].sum() - results_df["S2_net_pnl"].sum()),
        "delta_tier1": float(results_df["S4_net_pnl"].sum() - results_df["S2_net_pnl"].sum()),
        "delta_tier1_u": float(results_df["S5_net_pnl"].sum() - results_df["S3_net_pnl"].sum()),
    }


def run_one_cell(df: pd.DataFrame, common_days: list, eta_rt: float, c: float) -> pd.DataFrame:
    rows = []
    for day in common_days:
        day_df = df[df["delivery_date"] == day]
        result = run_backtest_for_day(day_df, eta_rt, c)
        result["delivery_date"] = day
        result["n_hours"] = len(day_df)
        rows.append(result)
    verify_structural_invariants(rows)

    out_rows = []
    for r in rows:
        row = {"delivery_date": r["delivery_date"], "n_hours": r["n_hours"], "oracle_pnl": r["oracle"]}
        for strat in STRATEGIES:
            row[f"{strat}_traded"] = r[strat]["traded"]
            row[f"{strat}_i"] = r[strat]["i"]
            row[f"{strat}_j"] = r[strat]["j"]
            row[f"{strat}_net_pnl"] = r[strat]["net_pnl"]
            row[f"{strat}_gross_pnl"] = r[strat]["gross_pnl"]
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def verify_base_case_reproduction(base_cell_results: pd.DataFrame, base_output_run_version: str, input_stem: str) -> None:
    """The gate: (0.85, 10) must reproduce the already-saved
    strategy_backtest_v1 EXACTLY -- same S0-S5 decision, same (i, j),
    same gross/net P&L, same oracle, same n_hours, same four aggregate
    deltas. An earlier version only checked traded flags and net P&L;
    two decisions could share identical net P&L while choosing
    different (i, j) pairs (e.g. on a day with more than one
    equally-profitable cycle), which the weaker check would have
    silently missed. Numeric comparisons use a 1e-9 tolerance --
    "exact" means economically identical, not byte-identical CSV text.
    """
    saved_path = REPO_ROOT / "outputs" / "strategy" / input_stem / base_output_run_version / "per_day_results.csv"
    if not saved_path.exists():
        raise FileNotFoundError(f"No saved base results at {saved_path}. Run run_strategy_backtest.py first.")
    saved = pd.read_csv(saved_path)
    saved["delivery_date"] = saved["delivery_date"].astype(str)
    computed = base_cell_results.copy()
    computed["delivery_date"] = computed["delivery_date"].astype(str)

    required_cols = [f"{s}_{suffix}" for s in STRATEGIES for suffix in ("i", "j")]
    missing = [c for c in required_cols if c not in saved.columns]
    if missing:
        raise ValueError(
            f"Saved base results at {saved_path} are missing {missing} -- this base run "
            f"predates the (i, j)-saving fix. Re-run run_strategy_backtest.py with a new "
            f"output_run_version to produce a base artifact this stronger reproduction "
            f"check can actually compare against."
        )

    merged = saved.merge(computed, on="delivery_date", how="outer", suffixes=("_saved", "_computed"), indicator=True)
    only_one_side = merged[merged["_merge"] != "both"]
    if len(only_one_side) > 0:
        raise AssertionError(
            f"BASE-CASE REPRODUCTION FAILED: {len(only_one_side)} day(s) present in only one of "
            f"the saved vs. recomputed results -- the common-day sets don't even match."
        )

    mismatches = []

    n_hours_diff = (merged["n_hours_saved"] - merged["n_hours_computed"]).abs()
    if (n_hours_diff > 0).any():
        mismatches.append(f"n_hours: {(n_hours_diff > 0).sum()} day(s) differ")

    oracle_diff = (merged["oracle_pnl_saved"] - merged["oracle_pnl_computed"]).abs()
    if (oracle_diff > 1e-9).any():
        mismatches.append(f"oracle_pnl: {(oracle_diff > 1e-9).sum()} day(s) differ")

    for strat in STRATEGIES:
        for col_suffix in ("traded", "i", "j"):
            saved_col, computed_col = f"{strat}_{col_suffix}_saved", f"{strat}_{col_suffix}_computed"
            # i/j can be NaN (no_trade) on both sides -- compare with NaN treated as equal.
            both_nan = merged[saved_col].isna() & merged[computed_col].isna()
            differs = (merged[saved_col] != merged[computed_col]) & ~both_nan
            if differs.any():
                mismatches.append(f"{strat}_{col_suffix}: {int(differs.sum())} day(s) differ")
        for col_suffix in ("net_pnl", "gross_pnl"):
            saved_col, computed_col = f"{strat}_{col_suffix}_saved", f"{strat}_{col_suffix}_computed"
            diff = (merged[saved_col] - merged[computed_col]).abs()
            if (diff > 1e-9).any():
                mismatches.append(f"{strat}_{col_suffix}: {(diff > 1e-9).sum()} day(s) differ")

    saved_deltas = compute_deltas(saved)
    computed_deltas = compute_deltas(computed)
    for key in saved_deltas:
        if abs(saved_deltas[key] - computed_deltas[key]) > 1e-6:
            mismatches.append(
                f"{key}: saved={saved_deltas[key]:.6f}, computed={computed_deltas[key]:.6f}"
            )

    if mismatches:
        raise AssertionError(
            "BASE-CASE REPRODUCTION FAILED -- the (eta_rt=0.85, c=10) cell does not exactly "
            "reproduce the already-saved strategy_backtest_v1 results. STOPPING before "
            "interpreting or computing any of the other eight sensitivity cells.\n" + "\n".join(mismatches)
        )


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    xgboost_run_version, uncertainty_run_version, tier1_run_version, base_output_run_version, output_run_version = resolve_run_args(sys.argv[1:])
    input_stem = "delu_features"

    out_dir = REPO_ROOT / "outputs" / "strategy" / input_stem / output_run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already contains results. Pass a new output_run_version.")

    logger.info("Loading frozen artifacts...")
    combined = load_combined_data(input_stem, xgboost_run_version, uncertainty_run_version, tier1_run_version)
    assert_no_holdout_access(combined["timestamp_utc"])

    day_info = build_common_evaluation_days(combined)
    common_days = day_info["common_days"]
    df = day_info["df_with_local_time"]
    logger.info("%d common evaluation days.", len(common_days))

    grid = list(product(ETA_RT_GRID, C_GRID))
    assert len(grid) == 9, f"Expected exactly 9 grid combinations, got {len(grid)}"

    print("\n" + "=" * 78)
    print(f"PRE-REGISTERED 3x3 STRATEGY SENSITIVITY GRID: {output_run_version}")
    print(f"eta_rt in {ETA_RT_GRID}, c in {C_GRID}")
    print("=" * 78)

    # CRITICAL GATE: run ONLY the base case first. The other eight cells
    # are not computed at all until this passes -- a literal, not just
    # documented, guarantee that a reproduction failure stops the
    # experiment before any other cell's computation is even attempted.
    logger.info("Running base case (eta_rt=%.2f, c=%d) first, per the reproduction gate...", *BASE_CASE)
    cell_results = {BASE_CASE: run_one_cell(df, common_days, *BASE_CASE)}

    logger.info("Verifying base-case (0.85, 10) reproduces %s exactly...", base_output_run_version)
    verify_base_case_reproduction(cell_results[BASE_CASE], base_output_run_version, input_stem)
    print(f"\nBASE-CASE REPRODUCTION CHECK PASSED -- (eta_rt=0.85, c=10) exactly matches "
          f"'{base_output_run_version}' (decisions, (i,j), gross/net P&L, oracle, and all "
          f"four aggregate deltas).")

    logger.info("Gate passed -- running the remaining 8 cells...")
    for eta_rt, c in grid:
        if (eta_rt, c) == BASE_CASE:
            continue
        logger.info("Running cell (eta_rt=%.2f, c=%d)...", eta_rt, c)
        cell_results[(eta_rt, c)] = run_one_cell(df, common_days, eta_rt, c)

    print(f"\nAll {len(cell_results)} cells computed, structural invariants passed for each.")

    # Extra cross-cell structural checks: common days identical across
    # every cell (they're all built from the same day_info, so this
    # should be true by construction -- confirmed explicitly anyway).
    n_days_per_cell = {k: len(v) for k, v in cell_results.items()}
    if len(set(n_days_per_cell.values())) != 1:
        raise AssertionError(f"Common day counts differ across cells: {n_days_per_cell}")
    for k, v in cell_results.items():
        assert (v["S0_net_pnl"] == 0.0).all(), f"S0 non-zero in cell {k}"
    print("Cross-cell checks passed: identical day counts, S0=0 in every cell.")

    grid_rows = []
    for (eta_rt, c), results_df in cell_results.items():
        deltas = compute_deltas(results_df)
        totals = {f"{s}_total_net_pnl": float(results_df[f"{s}_net_pnl"].sum()) for s in STRATEGIES}
        grid_rows.append({"eta_rt": eta_rt, "c": c, **totals, **deltas})
    grid_summary = pd.DataFrame(grid_rows).sort_values(["eta_rt", "c"]).reset_index(drop=True)

    print("\n" + "=" * 78)
    print("3x3 GRID SUMMARY (all nine cells)")
    print("=" * 78)
    print(grid_summary.to_string(index=False))

    print("\nSign counts across all 9 pre-registered cells (all four deltas were "
          "pre-registered together, not just delta_forecast):")
    for delta_name in ("delta_forecast", "delta_uncertainty", "delta_tier1", "delta_tier1_u"):
        signs = grid_summary[delta_name].apply(lambda x: "+" if x > 0 else ("-" if x < 0 else "0"))
        print(f"  {delta_name}: + {sum(signs=='+')}/9  | - {sum(signs=='-')}/9  | 0 {sum(signs=='0')}/9")

    out_dir.mkdir(parents=True, exist_ok=True)
    grid_summary.to_csv(out_dir / "sensitivity_grid_summary.csv", index=False)
    for (eta_rt, c), results_df in cell_results.items():
        results_df.to_csv(out_dir / f"per_day_results_eta{eta_rt}_c{c}.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "xgboost_run_version": xgboost_run_version,
        "uncertainty_run_version": uncertainty_run_version,
        "tier1_uncertainty_run_version": tier1_run_version,
        "base_output_run_version": base_output_run_version,
        "output_run_version": output_run_version,
        "eta_rt_grid": ETA_RT_GRID, "c_grid": C_GRID,
        "base_case_reproduction": "PASSED",
        "n_common_days": len(common_days),
        "holdout_used": False,
    }
    with open(out_dir / "sensitivity_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved 3x3 grid + manifest to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
