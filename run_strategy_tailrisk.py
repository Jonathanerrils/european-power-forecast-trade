"""Run locally: python run_strategy_tailrisk.py <backtest_run_version> <output_run_version>

Example:
  python run_strategy_tailrisk.py strategy_backtest_v2 strategy_tailrisk_v1

READ-ONLY tail-risk layer. Reads ONLY the already-frozen per-day net
P&L from <backtest_run_version>/per_day_results.csv -- never
recomputes a decision, never touches 2026, never re-tunes any strategy
parameter after seeing a VaR/ES number.

Loss is defined exactly once, here, and nowhere else:

    L_D,s = -Pi_D,s

This is a DIFFERENT quantity from uncertainty_selected_v1's price-
forecast-residual intervals (P - P_hat) -- that layer quantifies
FORECAST error; this layer quantifies STRATEGY loss, after position
sizing, efficiency, and degradation cost are already baked into
Pi_D,s. The two must never be conflated (see README "Uncertainty
quantification" scope-boundary note and docs/economic_contract_v1.md's
"Explicitly deferred" section, which named exactly this layer and
exactly this boundary before any tail-risk code existed).

PRE-REGISTERED confidence levels, fixed before any VaR/ES number is
computed: 95% and 99% -- the two standard risk levels, not chosen
after looking at results. Method: empirical/historical VaR and ES
(the simplest, most defensible baseline given the development
sample's size) -- a single pre-specified model, not a search over
candidate risk models.

HONEST ABOUT SAMPLE SIZE, not hidden: at n=1,081 common development
days, 99% VaR rests on ~11 tail observations. This script reports the
exact tail count next to every VaR/ES figure and flags any estimate
resting on fewer than 20 tail observations as LOW PRECISION -- it does
not silently present a thin-sample estimate with the same apparent
confidence as a well-supported one.

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

from run_strategy_backtest import STRATEGIES
from run_strategy_decomposition import verify_source_provenance
from src.utils import load_config, setup_logging, REPO_ROOT

# PRE-REGISTERED, fixed before this script has ever been run against
# real data -- do not add, remove, or select after seeing a result.
CONFIDENCE_LEVELS = [0.95, 0.99]
LOW_PRECISION_TAIL_THRESHOLD = 20


def resolve_run_args(args: list) -> tuple:
    if len(args) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python run_strategy_tailrisk.py <backtest_run_version> <output_run_version>\n\n"
            "Example:\n"
            "  python run_strategy_tailrisk.py strategy_backtest_v2 strategy_tailrisk_v1"
        )
    return tuple(args)


def load_backtest_pnl(input_stem: str, backtest_run_version: str) -> pd.DataFrame:
    path = REPO_ROOT / "outputs" / "strategy" / input_stem / backtest_run_version / "per_day_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_strategy_backtest.py first.")
    df = pd.read_csv(path)
    required = ["delivery_date"] + [f"{s}_net_pnl" for s in STRATEGIES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"'{backtest_run_version}' is missing required column(s): {missing}")
    return df


def validate_pnl_data(per_day: pd.DataFrame, holdout_boundary: str = "2026-01-01") -> None:
    """Fails closed before any VaR/ES is computed -- the same
    discipline already established in run_strategy_decomposition.py.
    pandas' quantile()/mean() silently skip NaN by default, which
    would otherwise let a missing P&L observation quietly vanish from
    a VaR/ES estimate rather than being caught.
    """
    problems = []

    dates = per_day["delivery_date"].astype(str)
    if dates.isna().any():
        problems.append("delivery_date contains missing value(s)")
    dup_days = dates[dates.duplicated()]
    if len(dup_days) > 0:
        problems.append(f"delivery_date contains {len(dup_days)} duplicate(s): {sorted(dup_days.unique())[:5]}")

    # Simple lexicographic comparison is correct here: delivery_date is
    # stored as an ISO "YYYY-MM-DD" string, which sorts identically to
    # chronological order.
    holdout_dates = dates[dates >= holdout_boundary]
    if len(holdout_dates) > 0:
        problems.append(
            f"{len(holdout_dates)} delivery date(s) at or after the {holdout_boundary} holdout "
            f"boundary -- tail-risk analysis must never touch the holdout."
        )

    for strat in STRATEGIES:
        pnl_col = per_day[f"{strat}_net_pnl"]
        if not np.isfinite(pnl_col.astype(float)).all():
            n_bad = int((~np.isfinite(pnl_col.astype(float))).sum())
            problems.append(f"{strat}_net_pnl contains {n_bad} non-finite value(s) (NaN/Inf)")

    if "S0_net_pnl" in per_day.columns and (per_day["S0_net_pnl"] != 0).any():
        n_bad = int((per_day["S0_net_pnl"] != 0).sum())
        problems.append(f"S0_net_pnl is nonzero on {n_bad} day(s) -- S0 (no-op benchmark) must always be exactly 0")

    if problems:
        raise ValueError(
            "INPUT VALIDATION FAILED -- refusing to compute VaR/ES on malformed persisted data:\n" +
            "\n".join(f"  - {p}" for p in problems)
        )


def compute_loss_series(per_day: pd.DataFrame, strategy: str) -> pd.Series:
    """L_D = -Pi_D for the given strategy. This is the ONLY place in
    this project's codebase that defines strategy-level loss -- every
    downstream risk computation must go through this function, never
    redefine loss inline.
    """
    return -per_day[f"{strategy}_net_pnl"]


def compute_empirical_var_es(loss_series: pd.Series, confidence_level: float) -> dict:
    """Empirical/historical VaR and ES on a LOSS series (positive =
    loss, matching compute_loss_series's sign convention).

    VaR at confidence level c: the c-quantile of the loss distribution
    -- the threshold such that a fraction c of days have a loss at or
    below it. Pandas' quantile() interpolates correctly regardless of
    ties, so VaR itself is NOT the buggy part -- a VaR of exactly 0
    for a heavily-abstaining strategy (S3, S5) is a genuinely correct
    reading of the distribution, not an artifact.

    ES (Expected Shortfall / CVaR): the mean loss over the exact worst
    m = (1-c)*n observation-equivalents, with fractional weight on the
    boundary observation -- NOT "every observation tied at or above
    the VaR threshold". The tied-threshold approach silently included
    hundreds of exact-zero no-trade days as "tail" observations for
    heavily-abstaining strategies, diluting ES toward the abstention
    value and making the strategy look far safer than its true worst-
    m-observations severity -- confirmed numerically (a synthetic
    reproduction of the real S3-shaped distribution gave ES=0.17 under
    the tied-threshold method vs. 1.38 under this exact-m method, an
    ~8x understatement) before this was rewritten, not assumed from
    the bug report alone.

    n_tail here is m itself (not an integer count of ties) -- the
    correct basis for the low-precision diagnostic, since m directly
    answers "how many effective observations does this ES rest on."
    """
    if not (0 < confidence_level < 1):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    n_total = len(loss_series)
    if n_total == 0:
        raise ValueError("Cannot compute VaR/ES on an empty loss series.")

    var = float(loss_series.quantile(confidence_level))

    m = (1 - confidence_level) * n_total
    sorted_desc = loss_series.sort_values(ascending=False).reset_index(drop=True)
    m_floor = int(np.floor(m))
    frac = m - m_floor
    worst_sum = float(sorted_desc.iloc[:m_floor].sum()) if m_floor > 0 else 0.0
    if frac > 0 and m_floor < n_total:
        worst_sum += frac * float(sorted_desc.iloc[m_floor])
    es = worst_sum / m if m > 0 else var

    return {
        "confidence_level": confidence_level,
        "var": var,
        "es": es,
        "n_total": n_total,
        "n_tail": m,
        "low_precision": m < LOW_PRECISION_TAIL_THRESHOLD,
    }


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    backtest_run_version, output_run_version = resolve_run_args(sys.argv[1:])
    input_stem = "delu_features"

    out_dir = REPO_ROOT / "outputs" / "risk" / input_stem / output_run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already contains results. Pass a new output_run_version.")

    logger.info("Verifying source provenance -- is '%s' actually the frozen primary base case?", backtest_run_version)
    verify_source_provenance(backtest_run_version, input_stem)
    print(f"Source provenance check PASSED for '{backtest_run_version}'.")

    logger.info("Loading frozen daily P&L from '%s' (read-only)...", backtest_run_version)
    per_day = load_backtest_pnl(input_stem, backtest_run_version)

    logger.info("Validating persisted P&L integrity before computing any VaR/ES...")
    validate_pnl_data(per_day)
    print("Input validation PASSED -- no malformed rows found.")
    n_days = len(per_day)

    print("\n" + "=" * 78)
    print(f"STRATEGY TAIL RISK: {output_run_version} (source: {backtest_run_version})")
    print(f"Loss defined as L_D = -Pi_D (strategy net P&L, NOT price-forecast residuals)")
    print(f"Development sample: {n_days} common days. Pre-registered confidence levels: "
          f"{[f'{c*100:.0f}%' for c in CONFIDENCE_LEVELS]}")
    print("=" * 78)

    rows = []
    for strat in STRATEGIES:
        loss = compute_loss_series(per_day, strat)
        print(f"\n--- {strat} ---")
        print(f"  Mean daily loss: EUR {loss.mean():.2f}  (negative = strategy typically profitable)")
        print(f"  Worst single-day loss: EUR {loss.max():.2f}")
        for cl in CONFIDENCE_LEVELS:
            result = compute_empirical_var_es(loss, cl)
            precision_flag = "  [LOW PRECISION -- fewer than 20 tail observations]" if result["low_precision"] else ""
            print(f"  VaR_{cl*100:.0f}: EUR {result['var']:.2f}   ES_{cl*100:.0f}: EUR {result['es']:.2f}   "
                  f"(effective tail size={result['n_tail']:.2f}/{result['n_total']}){precision_flag}")
            rows.append({"strategy": strat, **result})

    print("\n" + "=" * 78)
    print("LIMITATIONS (stated explicitly, not left implicit):")
    print("  - Empirical/historical method only -- no conditional (e.g. GARCH-based) or")
    print("    EVT/POT model included in this run. A pre-specified challenger could be added")
    print("    later as an explicitly separate, labeled comparison -- not a replacement chosen")
    print("    after seeing which one looks better.")
    print("  - No block-aware resampling for significance testing yet -- daily losses may")
    print("    cluster in time (as this project's own uncertainty-coverage work already showed")
    print("    misses do), which a naive i.i.d. treatment would understate.")
    print("  - Backtesting (Kupiec/Christoffersen coverage tests) not yet built -- this run")
    print("    reports point estimates only, not their own calibration.")
    print("  - Development sample only (2023-2025); 2026 holdout untouched.")
    print("=" * 78)

    results_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "var_es_results.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backtest_run_version": backtest_run_version,
        "output_run_version": output_run_version,
        "confidence_levels": CONFIDENCE_LEVELS,
        "method": "empirical_historical",
        "n_common_days": n_days,
        "low_precision_tail_threshold": LOW_PRECISION_TAIL_THRESHOLD,
        "holdout_used": False,
    }
    with open(out_dir / "tailrisk_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved VaR/ES results + manifest to {out_dir}")


if __name__ == "__main__":
    main()
