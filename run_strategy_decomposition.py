"""Run locally: python run_strategy_decomposition.py <strategy_backtest_run_version> <output_run_version>

Example:
  python run_strategy_decomposition.py strategy_backtest_v2 strategy_decomposition_v1

Decomposes the S2<->S3 (Full point vs. Full uncertainty-aware) and
S4<->S5 (Tier-1 point vs. Tier-1 uncertainty-aware) economic gaps into
interpretable components, per docs/economic_contract_v1.md's own
caution against promoting a plausible-sounding aggregate-number
narrative ("uncertainty loses because it's too selective") to a causal
claim without actually proving the decomposition.

For every common day, classified into exactly one of five categories:
  1. both_abstain
  2. point_trades_uncertainty_abstains  (profits forgone / losses avoided live here)
  3. point_abstains_uncertainty_trades  (expected near-zero -- see below)
  4. both_trade_same_pair               (must contribute exactly zero to the gap)
  5. both_trade_different_pair          (the pair-selection effect)

RECONCILIATION IS AN ASSERTION, NOT JUST A REPORTED NUMBER: the
identity
    Delta_Pi_U = -profits_forgone + losses_avoided + pair_selection_effect + reverse_category_effect
(with same_pair_difference required to equal exactly zero as a
SEPARATE structural invariant, not folded into the identity above)
must hold EXACTLY (to floating-point tolerance) against the actual
sum(Pi_uncertainty - Pi_point) across all common days. If it doesn't,
the decomposition itself has a bug and nothing downstream should be
trusted. An earlier version of this docstring omitted the
reverse-category and same-pair terms from the stated identity even
though the code always computed and checked them -- fixed here so the
specification and implementation say exactly the same thing.

FAILS CLOSED on malformed persisted input, not just missing columns:
every *_traded value must be a genuine boolean (not a string, not NaN
-- Python's bool() coerces both "False" and float('nan') to True,
which would silently corrupt the classification), every *_net_pnl
must be finite, traded days must have both (i, j) present with i<j,
no-trade days must have neither and exactly zero P&L, and delivery_date
must be unique. See validate_backtest_results() -- this matters because
a NaN net_pnl would otherwise be silently dropped by pandas' default
sum() behavior, letting a missing economic observation masquerade as
an exact reconciliation of zero.

PROVENANCE CHECK: verifies the source run's own manifest shows the
frozen primary parameters (eta_rt=0.85, c=10), holdout_used=false, and
structural_invariants=PASSED before decomposing it -- this
decomposition is specifically meant to explain the frozen base-case
result, not an arbitrary sensitivity cell someone points it at later.

Diagnostic, not a hard invariant: category 3 (point abstains,
uncertainty trades) is expected to be empty or near-empty, since the
uncertainty-aware score is a pointwise-dominated version of the point
score whenever the rolling residual quantile's q10 offset is <= 0 and
q90 offset is >= 0 (verified numerically, 5,000 synthetic trials, zero
violations, before writing this script) -- but this is an EMPIRICAL
property of the real residual distribution, not a mathematical
certainty, so a nonzero count here is reported, not asserted to be
zero.

Requires the input run to have (i, j) persisted -- i.e. a schema-
extended run (strategy_backtest_v2 or later), not the original
strategy_backtest_v1.

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

from src.utils import load_config, setup_logging, REPO_ROOT

RECONCILIATION_TOLERANCE = 1e-6


def resolve_run_args(args: list) -> tuple:
    if len(args) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python run_strategy_decomposition.py <strategy_backtest_run_version> <output_run_version>\n\n"
            "Example:\n"
            "  python run_strategy_decomposition.py strategy_backtest_v2 strategy_decomposition_v1"
        )
    return tuple(args)


def load_backtest_results(input_stem: str, backtest_run_version: str) -> pd.DataFrame:
    path = REPO_ROOT / "outputs" / "strategy" / input_stem / backtest_run_version / "per_day_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_strategy_backtest.py for '{backtest_run_version}' first.")
    df = pd.read_csv(path)
    required = ["delivery_date", "n_hours"] + [
        f"{s}_{suffix}" for s in ("S2", "S3", "S4", "S5") for suffix in ("traded", "i", "j", "net_pnl")
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{backtest_run_version}' is missing {missing} -- this decomposition requires (i, j) "
            f"persisted, i.e. a schema-extended run (strategy_backtest_v2 or later), not the "
            f"original strategy_backtest_v1."
        )
    return df


def validate_backtest_results(df: pd.DataFrame, pnl_tolerance: float = 1e-9) -> None:
    """Fails closed on malformed persisted input, not just missing
    columns. Every check here corresponds to a specific silent-failure
    mode confirmed by direct testing before this function was written:
    a NaN net_pnl is silently skipped by pandas' default sum(),
    producing a false "EXACT" reconciliation of zero for what should
    be a missing economic observation; a string "False" or a NaN in a
    *_traded column both evaluate True under Python's bool(), silently
    corrupting the five-category classification; a fractional or
    out-of-range (i, j) would pass a bare i<j check while not actually
    corresponding to a real hourly delivery interval; a non-numeric
    persisted (i, j) would otherwise raise an uncontrolled TypeError
    from downstream arithmetic (confirmed: '>=' not supported between
    'str' and 'int') rather than a clear validation error; an n_hours
    value outside {23, 24, 25} cannot correspond to a genuine DE-LU/
    Berlin delivery day (spring DST / normal / fall DST are the only
    three valid lengths).
    """
    problems = []

    if df["delivery_date"].isna().any():
        problems.append("delivery_date contains missing value(s)")
    dup_days = df["delivery_date"][df["delivery_date"].duplicated()]
    if len(dup_days) > 0:
        problems.append(f"delivery_date contains {len(dup_days)} duplicate(s): {sorted(dup_days.unique())[:5]}")

    # n_hours must be a real delivery-day length for DE-LU/Berlin:
    # exactly 23, 24, or 25 hours (DST spring-forward / normal / fall-back).
    # Coerce first so malformed strings fail closed with a controlled
    # validation error rather than raising from arithmetic later.
    n_hours_numeric = pd.to_numeric(df["n_hours"], errors="coerce")
    bad_n_hours = (
        ~np.isfinite(n_hours_numeric)
        | (n_hours_numeric % 1 != 0)
        | ~n_hours_numeric.isin([23, 24, 25])
    )
    if bad_n_hours.any():
        problems.append(
            f"n_hours contains {int(bad_n_hours.sum())} invalid value(s); "
            f"expected integer 23, 24, or 25"
        )

    for strat in ("S2", "S3", "S4", "S5"):
        traded_col = df[f"{strat}_traded"]
        if not traded_col.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            bad_types = sorted(set(type(v).__name__ for v in traded_col if not isinstance(v, (bool, np.bool_))))
            problems.append(f"{strat}_traded contains non-boolean value(s) (types: {bad_types})")
            continue  # skip the traded-dependent checks below for this strategy; type itself is already wrong

        pnl_col = df[f"{strat}_net_pnl"]
        if not np.isfinite(pnl_col.astype(float)).all():
            n_bad = int((~np.isfinite(pnl_col.astype(float))).sum())
            problems.append(f"{strat}_net_pnl contains {n_bad} non-finite value(s) (NaN/Inf)")

        i_col, j_col = df[f"{strat}_i"], df[f"{strat}_j"]
        traded_mask = traded_col.astype(bool)

        traded_missing_ij = traded_mask & (i_col.isna() | j_col.isna())
        if traded_missing_ij.any():
            problems.append(f"{strat}: {int(traded_missing_ij.sum())} traded day(s) missing (i, j)")

        no_trade_has_ij = (~traded_mask) & (i_col.notna() | j_col.notna())
        if no_trade_has_ij.any():
            problems.append(f"{strat}: {int(no_trade_has_ij.sum())} no-trade day(s) have a non-null (i, j)")

        # Convert persisted indices explicitly before any arithmetic. This
        # turns malformed object/string values into NaN so they are
        # reported through INPUT VALIDATION FAILED instead of leaking a
        # raw TypeError from a str-vs-int comparison downstream.
        i_numeric = pd.to_numeric(i_col, errors="coerce")
        j_numeric = pd.to_numeric(j_col, errors="coerce")

        traded_present_ij = traded_mask & i_col.notna() & j_col.notna()
        non_numeric_ij = traded_present_ij & (i_numeric.isna() | j_numeric.isna())
        if non_numeric_ij.any():
            problems.append(f"{strat}: {int(non_numeric_ij.sum())} traded day(s) with non-numeric (i, j)")

        valid_ij = traded_mask & i_numeric.notna() & j_numeric.notna()
        if valid_ij.any():
            bad_order = valid_ij & (i_numeric >= j_numeric)
            if bad_order.any():
                problems.append(f"{strat}: {int(bad_order.sum())} traded day(s) with i >= j")

            # Integrality: a fractional index like 2.5 can't correspond to
            # a real hourly delivery interval, and would otherwise pass
            # the i<j check silently.
            not_integer = valid_ij & (
                (i_numeric % 1 != 0) | (j_numeric % 1 != 0)
            )
            if not_integer.any():
                problems.append(f"{strat}: {int(not_integer.sum())} traded day(s) with non-integer (i, j)")

            # Bounds: i, j must lie within [0, n_hours) for that specific
            # day -- an out-of-range index (e.g. j=999 on a 24-hour day)
            # would otherwise pass a bare i<j comparison. Uses the
            # already-coerced numeric series throughout so a malformed
            # persisted type can't trigger an object-dtype comparison error.
            out_of_bounds = valid_ij & (
                (i_numeric < 0) | (j_numeric >= n_hours_numeric)
            )
            if out_of_bounds.any():
                problems.append(f"{strat}: {int(out_of_bounds.sum())} traded day(s) with (i, j) outside [0, n_hours)")

        no_trade_nonzero_pnl = (~traded_mask) & pnl_col.notna() & (pnl_col.abs() > pnl_tolerance)
        if no_trade_nonzero_pnl.any():
            problems.append(f"{strat}: {int(no_trade_nonzero_pnl.sum())} no-trade day(s) with nonzero net_pnl")

    if problems:
        raise ValueError(
            "INPUT VALIDATION FAILED -- refusing to decompose malformed persisted data:\n" +
            "\n".join(f"  - {p}" for p in problems)
        )


def verify_source_provenance(backtest_run_version: str, input_stem: str) -> dict:
    """This decomposition is specifically meant to explain the frozen
    base-case result -- confirms the source run's own manifest actually
    IS that base case (eta_rt=0.85, c=10, holdout untouched, structural
    invariants passed) before decomposing it, rather than assuming
    whoever points this script at a run_version did so correctly.
    """
    manifest_path = REPO_ROOT / "outputs" / "strategy" / input_stem / backtest_run_version / "strategy_backtest_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path} -- cannot verify source provenance.")
    with open(manifest_path) as f:
        manifest = json.load(f)

    problems = []
    if manifest.get("eta_rt") != 0.85:
        problems.append(f"eta_rt={manifest.get('eta_rt')}, expected 0.85 (the frozen primary case)")
    if manifest.get("c") != 10.0 and manifest.get("c") != 10:
        problems.append(f"c={manifest.get('c')}, expected 10 (the frozen primary case)")
    if manifest.get("holdout_used") is not False:
        problems.append(f"holdout_used={manifest.get('holdout_used')}, expected False")
    if manifest.get("structural_invariants") != "PASSED":
        problems.append(f"structural_invariants={manifest.get('structural_invariants')}, expected 'PASSED'")

    if "parent_run" in manifest:
        if manifest.get("change_type") != "schema_extension_only":
            problems.append(
                f"has 'parent_run' but change_type={manifest.get('change_type')!r}, "
                f"expected 'schema_extension_only'"
            )
        if manifest.get("legacy_result_reproduction") != "PASSED":
            problems.append(
                f"has 'parent_run' but legacy_result_reproduction={manifest.get('legacy_result_reproduction')!r}, "
                f"expected 'PASSED'"
            )

    if problems:
        raise ValueError(
            f"SOURCE PROVENANCE CHECK FAILED for '{backtest_run_version}' -- this does not appear "
            f"to be the frozen primary base case:\n" + "\n".join(f"  - {p}" for p in problems)
        )

    if "parent_run" not in manifest:
        print(f"  NOTE: '{backtest_run_version}' has no 'parent_run' lineage field -- treating it as a "
              f"first-ever run rather than a verified schema-extension descendant. If this was meant to "
              f"be a schema-extension of an earlier run, confirm that was actually verified.")

    return manifest


def classify_day(row: pd.Series, point_strat: str, uncertainty_strat: str) -> str:
    """Classifies one day into exactly one of the five decomposition
    categories, based on each strategy's traded flag and, when both
    traded, whether they chose the same (i, j) pair.
    """
    point_traded = bool(row[f"{point_strat}_traded"])
    unc_traded = bool(row[f"{uncertainty_strat}_traded"])
    if not point_traded and not unc_traded:
        return "both_abstain"
    if point_traded and not unc_traded:
        return "point_trades_uncertainty_abstains"
    if not point_traded and unc_traded:
        return "point_abstains_uncertainty_trades"
    # both traded
    same_pair = (row[f"{point_strat}_i"] == row[f"{uncertainty_strat}_i"]) and (
        row[f"{point_strat}_j"] == row[f"{uncertainty_strat}_j"]
    )
    return "both_trade_same_pair" if same_pair else "both_trade_different_pair"


def decompose(df: pd.DataFrame, point_strat: str, uncertainty_strat: str) -> dict:
    """Returns the full decomposition for ONE point-vs-uncertainty-aware
    pair (S2 vs S3, or S4 vs S5). Raises AssertionError if the
    reconciliation doesn't hold exactly.
    """
    categories = df.apply(lambda row: classify_day(row, point_strat, uncertainty_strat), axis=1)
    df = df.copy()
    df["_category"] = categories

    point_pnl = df[f"{point_strat}_net_pnl"]
    unc_pnl = df[f"{uncertainty_strat}_net_pnl"]

    forgone_mask = df["_category"] == "point_trades_uncertainty_abstains"
    profits_forgone = float(point_pnl[forgone_mask].clip(lower=0).sum())
    losses_avoided = float((-point_pnl[forgone_mask].clip(upper=0)).sum())

    same_pair_mask = df["_category"] == "both_trade_same_pair"
    same_pair_diff = float((unc_pnl[same_pair_mask] - point_pnl[same_pair_mask]).sum())

    diff_pair_mask = df["_category"] == "both_trade_different_pair"
    pair_selection_effect = float((unc_pnl[diff_pair_mask] - point_pnl[diff_pair_mask]).sum())

    reverse_mask = df["_category"] == "point_abstains_uncertainty_trades"
    n_reverse = int(reverse_mask.sum())
    reverse_effect = float((unc_pnl[reverse_mask] - point_pnl[reverse_mask]).sum())  # point_pnl is 0 here by definition

    total_gap = float((unc_pnl - point_pnl).sum())
    reconciled = -profits_forgone + losses_avoided + pair_selection_effect + same_pair_diff + reverse_effect

    if abs(reconciled - total_gap) > RECONCILIATION_TOLERANCE:
        raise AssertionError(
            f"RECONCILIATION FAILED for {point_strat}<->{uncertainty_strat}: "
            f"components sum to {reconciled:.6f} but actual total gap is {total_gap:.6f} "
            f"(difference {reconciled - total_gap:.6f}). The decomposition itself has a bug."
        )
    if abs(same_pair_diff) > RECONCILIATION_TOLERANCE:
        raise AssertionError(
            f"STRUCTURAL VIOLATION: 'both_trade_same_pair' days contributed a nonzero P&L "
            f"difference ({same_pair_diff:.6f}) -- identical (i, j) on the same day with the "
            f"same actual prices must produce identical realized P&L regardless of which rule "
            f"chose the pair."
        )

    return {
        "point_strategy": point_strat,
        "uncertainty_strategy": uncertainty_strat,
        "n_days": len(df),
        "n_both_abstain": int((df["_category"] == "both_abstain").sum()),
        "n_point_trades_uncertainty_abstains": int(forgone_mask.sum()),
        "n_point_abstains_uncertainty_trades": n_reverse,
        "n_both_trade_same_pair": int(same_pair_mask.sum()),
        "n_both_trade_different_pair": int(diff_pair_mask.sum()),
        "profits_forgone": profits_forgone,
        "losses_avoided": losses_avoided,
        "pair_selection_effect": pair_selection_effect,
        "reverse_category_effect": reverse_effect,
        "total_gap": total_gap,
        "reconciliation": "EXACT",
        "diagnostic_reverse_category_nonzero": n_reverse > 0,
    }


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    backtest_run_version, output_run_version = resolve_run_args(sys.argv[1:])
    input_stem = "delu_features"

    out_dir = REPO_ROOT / "outputs" / "strategy" / input_stem / output_run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already contains results. Pass a new output_run_version.")

    logger.info("Verifying source provenance -- is '%s' actually the frozen primary base case?", backtest_run_version)
    verify_source_provenance(backtest_run_version, input_stem)
    print(f"Source provenance check PASSED for '{backtest_run_version}'.")

    logger.info("Loading '%s'...", backtest_run_version)
    df = load_backtest_results(input_stem, backtest_run_version)

    logger.info("Validating persisted data integrity before decomposing...")
    validate_backtest_results(df)
    print("Input validation PASSED -- no malformed rows found.")

    results = {}
    for point_strat, uncertainty_strat in [("S2", "S3"), ("S4", "S5")]:
        logger.info("Decomposing %s <-> %s...", point_strat, uncertainty_strat)
        results[f"{point_strat}_{uncertainty_strat}"] = decompose(df, point_strat, uncertainty_strat)

    print("\n" + "=" * 78)
    print(f"STRATEGY DECOMPOSITION: {output_run_version} (source: {backtest_run_version})")
    print("=" * 78)
    for key, r in results.items():
        print(f"\n--- {r['point_strategy']} <-> {r['uncertainty_strategy']} ---")
        print(f"  Days: both abstain={r['n_both_abstain']}, "
              f"{r['point_strategy']} trades/{r['uncertainty_strategy']} abstains={r['n_point_trades_uncertainty_abstains']}, "
              f"{r['point_strategy']} abstains/{r['uncertainty_strategy']} trades={r['n_point_abstains_uncertainty_trades']}"
              f"{'  [DIAGNOSTIC: nonzero, worth inspecting]' if r['diagnostic_reverse_category_nonzero'] else ''}, "
              f"both trade same pair={r['n_both_trade_same_pair']}, both trade different pair={r['n_both_trade_different_pair']}")
        print(f"  Profits forgone (by abstention):  EUR {r['profits_forgone']:.2f}")
        print(f"  Losses avoided (by abstention):   EUR {r['losses_avoided']:.2f}")
        print(f"  Pair-selection effect:            EUR {r['pair_selection_effect']:.2f}")
        if r['reverse_category_effect'] != 0:
            print(f"  Reverse-category effect:          EUR {r['reverse_category_effect']:.2f}")
        print(f"  RECONCILED total gap:              EUR {r['total_gap']:.2f}  (reconciliation: {r['reconciliation']})")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "decomposition_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backtest_run_version": backtest_run_version,
        "output_run_version": output_run_version,
        "reconciliation_tolerance": RECONCILIATION_TOLERANCE,
        "holdout_used": False,
    }
    with open(out_dir / "decomposition_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved decomposition + manifest to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
