"""Run locally: python scripts/compare_eda_runs.py <old_run_version> <new_run_version> [input_stem] [scope]

Example:
  python scripts/compare_eda_runs.py development_pre_a03fix development_a03fix_v1

Diffs two versioned run_eda.py output directories' summary CSVs and
prints a structured old-vs-new comparison, rather than requiring a
human to eyeball two console dumps side by side.

Reads directly from outputs/eda/<input_stem>/<scope>/<run_version>/tables/
-- both run_versions must already exist (run_eda.py for each first).
Does not recompute or refit anything; this is purely a diff of
already-saved evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils import REPO_ROOT

# Key scalar/small-table comparisons worth surfacing explicitly, rather
# than dumping every cell of every CSV -- these are the specific
# quantities the project has repeatedly cared about (see README
# "Uncertainty quantification" / EDA sections for why each matters).
KEY_FILES = [
    "price_distribution_summary.csv",
    "fundamentals_summary.csv",
    "fundamentals_missingness.csv",
    "price_by_year.csv",
    "price_by_market_design_regime.csv",
    "residual_load_price_corr_by_year.csv",
]


def resolve_run_args(args: list) -> tuple:
    if len(args) not in (2, 3, 4):
        raise SystemExit(
            "Usage:\n"
            "  python scripts/compare_eda_runs.py <old_run_version> <new_run_version> "
            "[input_stem] [scope]\n\n"
            "Example:\n"
            "  python scripts/compare_eda_runs.py development_pre_a03fix development_a03fix_v1"
        )
    old_run = args[0]
    new_run = args[1]
    input_stem = args[2] if len(args) > 2 else "delu_features"
    scope = args[3] if len(args) > 3 else "development"
    return old_run, new_run, input_stem, scope


def load_eda_tables(input_stem: str, scope: str, run_version: str) -> dict:
    table_dir = REPO_ROOT / "outputs" / "eda" / input_stem / scope / run_version / "tables"
    if not table_dir.exists():
        raise FileNotFoundError(
            f"No EDA output found at {table_dir}. Run "
            f"'python run_eda.py <input_filename> {run_version}' first."
        )
    tables = {}
    for filename in KEY_FILES:
        path = table_dir / filename
        if path.exists():
            tables[filename] = pd.read_csv(path, index_col=0)
    return tables


def diff_scalar_table(old_df: pd.DataFrame, new_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """For tables where index+columns line up between runs (same row/
    column labels), computes new-old and (new-old)/old for every
    aligned numeric cell. Cells that don't exist in both runs (e.g. a
    year present in one run but not the other) are reported as NaN
    rather than silently dropped.
    """
    aligned_old, aligned_new = old_df.align(new_df, join="outer")
    numeric_old = aligned_old.select_dtypes(include="number")
    numeric_new = aligned_new.select_dtypes(include="number")
    common_cols = [c for c in numeric_new.columns if c in numeric_old.columns]

    rows = []
    for col in common_cols:
        for idx in numeric_new.index:
            old_val = numeric_old.loc[idx, col] if idx in numeric_old.index else None
            new_val = numeric_new.loc[idx, col] if idx in numeric_new.index else None
            if pd.isna(old_val) or pd.isna(new_val):
                continue
            diff = new_val - old_val
            rel = diff / old_val if old_val != 0 else float("nan")
            rows.append({"table": label, "row": idx, "column": col, "old": old_val, "new": new_val,
                         "diff": diff, "pct_change": rel * 100})
    return pd.DataFrame(rows)


def main():
    old_run, new_run, input_stem, scope = resolve_run_args(sys.argv[1:])

    print("\n" + "=" * 78)
    print(f"EDA COMPARISON: '{old_run}' (old) -> '{new_run}' (new)")
    print("=" * 78)

    old_tables = load_eda_tables(input_stem, scope, old_run)
    new_tables = load_eda_tables(input_stem, scope, new_run)

    common_files = sorted(set(old_tables.keys()) & set(new_tables.keys()))
    missing_in_old = sorted(set(new_tables.keys()) - set(old_tables.keys()))
    missing_in_new = sorted(set(old_tables.keys()) - set(new_tables.keys()))
    if missing_in_old:
        print(f"\nFiles present in '{new_run}' but not '{old_run}' (skipped): {missing_in_old}")
    if missing_in_new:
        print(f"\nFiles present in '{old_run}' but not '{new_run}' (skipped): {missing_in_new}")

    all_diffs = []
    for filename in common_files:
        diff = diff_scalar_table(old_tables[filename], new_tables[filename], filename)
        if diff.empty:
            continue
        all_diffs.append(diff)
        print(f"\n--- {filename} ---")
        # Sort by absolute pct_change so the most-changed cells surface first.
        diff_sorted = diff.reindex(diff["pct_change"].abs().sort_values(ascending=False).index)
        print(diff_sorted.head(15).to_string(index=False))

    if all_diffs:
        combined = pd.concat(all_diffs, ignore_index=True)
        out_dir = REPO_ROOT / "outputs" / "eda" / input_stem / scope / f"comparison_{old_run}_vs_{new_run}"
        out_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_dir / "eda_comparison.csv", index=False)
        print(f"\nSaved full comparison to {out_dir / 'eda_comparison.csv'}")
    else:
        print("\nNo comparable numeric cells found between the two runs.")
    print("=" * 78)


if __name__ == "__main__":
    main()
