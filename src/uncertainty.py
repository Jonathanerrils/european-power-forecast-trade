"""Rolling-window empirical-residual quantile forecasting (spec's
`uncertainty` config section: quantiles=[0.1, 0.5, 0.9],
residual_window_days=180).

Design, and why it's built this way rather than a separate quantile
regression model:

The project's four chronological folds (fold_1: val 2023, fold_2: val
2024, fold_3: val Jan-Sep 2025, regime_stress_test: val Oct-Dec 2025)
have CONTIGUOUS validation windows (each fold's val_end equals the
next fold's val_start -- see src/splits.py). Every prediction in every
fold is genuinely out-of-sample (made by a model trained only on data
strictly before that fold's validation window). Concatenating all four
folds' xgboost_full predictions therefore produces one continuous,
walk-forward, entirely out-of-sample residual timeline from
2023-01-01 through 2025-12-31 -- exactly the kind of series a rolling
residual-quantile method needs, built for free from work already done,
not a new backtest.

CRITICAL LEAKAGE RULE, tested explicitly below: the quantile interval
attached to the prediction for timestamp t must be built ONLY from
residuals at timestamps STRICTLY BEFORE t (a trailing window, closed
on the left, excluding t itself). This mirrors the exact discipline
src/features.py already applies to price lags -- an interval that used
t's own residual (or any later one) would leak the very error it's
supposed to be quantifying.

Two distinct, separately-scoped operations, not one function trying to
do both:
  1. compute_rolling_residual_quantiles(): for CALIBRATION TESTING --
     "as of each historical timestamp, what would the trailing-window
     residual quantiles have been?" Operates on residual_series's own
     timestamps only.
  2. latest_residual_quantile_offsets(): for ATTACHING AN INTERVAL TO A
     GENUINELY NEW FORECAST (e.g. the holdout) -- returns the single
     most recent window's quantile offsets, which the caller adds to
     any new point forecast. Deliberately NOT a rolling recomputation
     against new timestamps: a new forecast's own residual is, by
     definition, not yet known, so there is nothing to roll.

STANDING CAVEAT, inherited from the whole project's open question: the
underlying target price series depends on auction_sequence == 1, which
is a documented, still-open assumption pending external EPEX
verification (see README Limitations, scripts/verify_auction_sequence.py).
Every function here operates on whatever price series it's given --
if that assumption is later revised, this uncertainty layer inherits
the correction automatically on the next reproduction, same as the
point forecasts do, but is not itself capable of detecting or
correcting the underlying assumption.
"""
from __future__ import annotations

import logging
from typing import List

import pandas as pd

logger = logging.getLogger("power_forecast.uncertainty")


def build_continuous_residual_series(
    fold_predictions: List[pd.DataFrame],
    target_col: str,
    pred_col: str,
    ts_col: str = "timestamp_utc",
    fold_names: List[str] = None,
) -> pd.DataFrame:
    """Concatenates multiple folds' prediction DataFrames (each with
    ts_col, target_col, pred_col) into one chronologically sorted
    series with `prediction` and `residual` (target - prediction)
    columns. Asserts the folds are genuinely contiguous/non-overlapping
    in time -- if two folds' timestamps overlapped, some rows would
    have two different "out-of-sample" residuals for the same instant,
    which would silently corrupt the rolling window below. Duplicate
    timestamps across folds are a structural sign something is wrong
    (e.g. a fold's val window boundary changed without the others
    changing to match), not something to average away.

    fold_names, if given (parallel to fold_predictions), adds a `fold`
    column tagging each row with which fold it came from. Deliberately
    an explicit, passed-in label -- not re-derived from the timestamp
    against src/splits.py's boundaries a second time, which would be
    exactly the kind of implicit reconstruction that has drifted out
    of sync with reality elsewhere in this project before.
    """
    if fold_names is not None and len(fold_names) != len(fold_predictions):
        raise ValueError(
            f"fold_names has {len(fold_names)} entries but fold_predictions has "
            f"{len(fold_predictions)} -- must be parallel lists."
        )

    frames = []
    for i, df in enumerate(fold_predictions):
        piece = df[[ts_col, target_col, pred_col]].rename(columns={pred_col: "prediction"}).copy()
        if fold_names is not None:
            piece["fold"] = fold_names[i]
        frames.append(piece)
    combined = pd.concat(frames, ignore_index=True)
    combined[ts_col] = pd.to_datetime(combined[ts_col], utc=True)
    dupes = combined[ts_col].duplicated(keep=False)
    if dupes.any():
        n_dupes = int(dupes.sum())
        raise ValueError(
            f"{n_dupes} timestamp(s) appear in more than one fold's predictions -- folds are "
            f"supposed to be contiguous and non-overlapping. Pooling them anyway would silently "
            f"let one timestamp's residual be computed from the wrong fold's model, or averaged "
            f"across two different out-of-sample predictions. Investigate the fold definitions "
            f"(src/splits.py) rather than deduping here."
        )
    combined = combined.sort_values(ts_col).reset_index(drop=True)
    combined["residual"] = combined[target_col] - combined["prediction"]
    return combined


def _infer_min_periods_rows(index: pd.DatetimeIndex, min_periods_days: int) -> int:
    if len(index) < 2:
        raise ValueError("Need at least 2 rows to infer observation frequency.")
    median_gap_hours = index.to_series().diff().dropna().dt.total_seconds().median() / 3600
    if median_gap_hours <= 0:
        raise ValueError("Could not infer a positive observation frequency from the timestamps.")
    return max(1, int(min_periods_days * 24 / median_gap_hours))


def compute_rolling_residual_quantiles(
    residual_series: pd.DataFrame,
    quantiles: List[float],
    window_days: int,
    min_periods_days: int,
    ts_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """For EVERY row in residual_series (its own timestamps -- this is
    the calibration-testing path, not a way to attach intervals to
    different, new timestamps), computes empirical quantiles of the
    `residual` column over the trailing window_days, EXCLUDING the
    current row's own timestamp (closed='left'). Returns
    (ts_col, prediction, forecast_qXX...) -- prediction + each
    quantile's residual offset, ready to compare against the actual
    target for calibration testing.

    min_periods_days guards the warm-up period: a quantile computed
    from too few residuals is not meaningfully "the 10th percentile",
    it's noise dressed up as a number. Rows without min_periods_days
    worth of trailing residuals get NaN quantile forecasts rather than
    a falsely confident interval from a handful of points.
    """
    if not (0 < min_periods_days <= window_days):
        raise ValueError(f"min_periods_days ({min_periods_days}) must be in (0, window_days={window_days}]")

    df = residual_series.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col).reset_index(drop=True)

    indexed = df.set_index(ts_col)
    min_periods_rows = _infer_min_periods_rows(indexed.index, min_periods_days)

    # closed='left': window for the row at time t is [t - window_days, t),
    # EXCLUDING t itself -- the leakage-safety guarantee this module exists
    # to provide.
    rolling = indexed["residual"].rolling(window=f"{window_days}D", closed="left", min_periods=min_periods_rows)

    result = pd.DataFrame({ts_col: df[ts_col], "prediction": df["prediction"]})
    for q in quantiles:
        offset = rolling.quantile(q).reset_index(drop=True)
        result[f"forecast_q{int(q*100)}"] = df["prediction"] + offset
    return result


def latest_residual_quantile_offsets(
    residual_series: pd.DataFrame,
    quantiles: List[float],
    window_days: int,
    min_periods_days: int,
    ts_col: str = "timestamp_utc",
) -> dict:
    """Returns the SINGLE most recent trailing-window's residual
    quantile offsets, as of the last timestamp in residual_series --
    the offsets to add to a genuinely NEW forecast (e.g. the first
    holdout prediction), whose own residual is by definition not yet
    known. This is NOT a rolling recomputation against new future
    timestamps; it's one fixed snapshot, taken as of "now" (the end of
    the available residual history).
    """
    quantile_forecasts = compute_rolling_residual_quantiles(
        residual_series, quantiles, window_days, min_periods_days, ts_col=ts_col
    )
    last_row = quantile_forecasts.iloc[-1]
    offsets = {}
    for q in quantiles:
        col = f"forecast_q{int(q*100)}"
        offsets[f"q{int(q*100)}"] = float(last_row[col] - last_row["prediction"])
    return offsets


def evaluate_interval_calibration_by_fold(
    df: pd.DataFrame,
    actual_col: str,
    lower_col: str,
    upper_col: str,
    nominal_coverage: float,
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Same calibration check as evaluate_interval_calibration(), but
    broken out per fold rather than pooled across all of them. A single
    pooled coverage number can look fine while hiding real regime
    differences -- e.g. this project already found that
    regime_stress_test (the newest regime, closest to the eventual
    holdout) has visibly different volatility than the earlier folds
    (see EDA: post-cutover std 44.35 vs pre-cutover std 95.14). Whether
    the SAME rolling residual quantiles are equally well-calibrated in
    that specific regime, not just on average, is exactly the kind of
    thing a pooled number can't tell you.

    Requires df to have a `fold_col` column (see build_continuous_residual_series's
    fold_names parameter) -- raises rather than silently returning a
    single-row "pooled" result if that column is missing, since a
    caller asking for a per-fold breakdown should get one or a clear
    error, not a quiet fallback to the pooled number.
    """
    if fold_col not in df.columns:
        raise ValueError(
            f"'{fold_col}' column not found -- build_continuous_residual_series must be called "
            f"with fold_names to enable a per-fold calibration breakdown."
        )
    rows = []
    for fold_name, group in df.groupby(fold_col, sort=False):
        calib = evaluate_interval_calibration(
            group[actual_col], group[lower_col], group[upper_col], nominal_coverage
        )
        calib[fold_col] = fold_name
        rows.append(calib)
    # Preserve the fold's first-appearance (chronological) order, not
    # groupby's default (alphabetical/hash) order -- reading "fold_1,
    # fold_2, fold_3, regime_stress_test" top to bottom should mean
    # something chronologically, not be an accident of sort order.
    order = list(dict.fromkeys(df[fold_col]))
    result = pd.DataFrame(rows).set_index(fold_col).loc[order].reset_index()
    return result


def evaluate_interval_calibration(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    nominal_coverage: float,
) -> dict:
    """Empirical coverage check: what fraction of actual values fall
    within [lower, upper]? Should be close to nominal_coverage if the
    quantile method is well-calibrated. Also reports the fraction below
    lower and above upper separately -- a poorly calibrated interval
    can have the RIGHT total coverage while being badly asymmetric
    (e.g. almost everything below lower, nothing above upper), which a
    single coverage number would hide.
    """
    valid = actual.notna() & lower.notna() & upper.notna()
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "empirical_coverage": None, "frac_below_lower": None, "frac_above_upper": None}
    a, lo, hi = actual[valid], lower[valid], upper[valid]
    within = (a >= lo) & (a <= hi)
    below = a < lo
    above = a > hi
    return {
        "n": n,
        "nominal_coverage": nominal_coverage,
        "empirical_coverage": float(within.mean()),
        "frac_below_lower": float(below.mean()),
        "frac_above_upper": float(above.mean()),
    }


def compute_interval_score(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    alpha: float,
) -> float:
    """Winkler/interval score for a (1-alpha)*100% prediction interval.
    LOWER is better. This is a PROPER scoring rule: it penalizes width
    directly (so a narrower interval scores better, all else equal)
    AND penalizes a miss proportionally to how far past the bound the
    true value falls (scaled by 2/alpha) -- an interval cannot improve
    its score by simply being wider (that raises the width penalty on
    every single row) or simply being narrower with more misses (that
    raises the miss penalty). This is specifically why it's the primary
    metric for the window-sensitivity comparison in
    run_uncertainty_sensitivity.py: coverage alone can be "solved" by
    widening the interval until it swallows everything, and width alone
    can be "solved" by narrowing until it's useless -- the interval
    score cannot be gamed by either move alone.

    IS_alpha(l, u, y) = (u - l) + (2/alpha)*(l-y)*1{y<l} + (2/alpha)*(y-u)*1{y>u}
    """
    valid = actual.notna() & lower.notna() & upper.notna()
    a, lo, hi = actual[valid], lower[valid], upper[valid]
    width = hi - lo
    below_penalty = (2 / alpha) * (lo - a).clip(lower=0)
    above_penalty = (2 / alpha) * (a - hi).clip(lower=0)
    return float((width + below_penalty + above_penalty).mean())


def find_common_evaluation_start(
    residual_series: pd.DataFrame,
    quantiles: List[float],
    candidate_configs: List[tuple],
    ts_col: str = "timestamp_utc",
) -> pd.Timestamp:
    """Returns the earliest timestamp at which EVERY candidate
    (window_days, min_periods_days) pair has a valid (non-NaN)
    quantile forecast -- i.e. the latest of each candidate's own
    warm-up end. Evaluating every candidate only from this timestamp
    onward puts them on a genuinely common row set.

    This is the SAME principle this project already enforces for
    point-model comparison (baseline_v1's guarantee that all six
    models are scored on an identical common_comparison_rows set) --
    not a new idea. Different candidate windows have different
    warm-up lengths by construction (min_periods_days scales with
    window_days), so without this correction, a shorter window is
    partly being scored on early-2023 data a longer window never gets
    a chance to be evaluated on at all -- confounding "which window is
    better" with "which period of data got included."
    """
    sorted_q = sorted(quantiles)
    lo_q = sorted_q[0]
    lower_col = f"forecast_q{int(lo_q*100)}"
    cutoffs = []
    for window_days, min_periods_days in candidate_configs:
        qf = compute_rolling_residual_quantiles(residual_series, quantiles, window_days, min_periods_days, ts_col)
        valid = qf.loc[qf[lower_col].notna(), ts_col]
        if valid.empty:
            raise ValueError(
                f"window_days={window_days}, min_periods_days={min_periods_days} produces zero valid "
                f"rows on this residual series -- cannot include it in a common-row comparison."
            )
        cutoffs.append(valid.min())
    return max(cutoffs)


def evaluate_window_candidate(
    residual_series: pd.DataFrame,
    quantiles: List[float],
    window_days: int,
    min_periods_days: int,
    target_col: str,
    fold_col: str = "fold",
    common_start: pd.Timestamp = None,
    ts_col: str = "timestamp_utc",
    return_per_row: bool = False,
) -> dict:
    """Full evaluation of ONE candidate window_days for the sensitivity
    experiment: pooled + per-fold calibration, sharpness (mean/median
    interval width), and the interval score -- the three dimensions a
    window choice must be judged on together, not just calibration
    alone (see compute_interval_score's docstring for why coverage
    alone is gameable).

    common_start, if given (see find_common_evaluation_start), restricts
    evaluation to rows AT OR AFTER that timestamp -- putting every
    candidate in a sensitivity comparison on an identical row set. Left
    as None by default so this function still works standalone (e.g. a
    single-candidate evaluation with no comparison to align against).

    return_per_row, if True, includes the full per-timestamp
    (ts_col, prediction, lower_col, upper_col) DataFrame in the
    returned dict under "per_row" -- needed by callers that must SAVE
    the actual quantile bounds for downstream consumption (e.g.
    run_uncertainty_tier1_robustness.py, so a later strategy backtest
    can read Tier-1's frozen OOS bounds per
    docs/economic_contract_v1.md's provenance rule, rather than
    recomputing them). Left False by default so existing callers (the
    window-sensitivity experiment, comparing 5 candidates) don't pay
    for and carry around per-row data they never use.
    """
    quantile_forecasts = compute_rolling_residual_quantiles(
        residual_series, quantiles, window_days, min_periods_days
    )
    merged = residual_series.merge(
        quantile_forecasts.drop(columns=["prediction"]), on="timestamp_utc", how="left"
    )
    if common_start is not None:
        merged = merged[merged[ts_col] >= common_start].reset_index(drop=True)

    sorted_q = sorted(quantiles)
    lo_q, hi_q = sorted_q[0], sorted_q[-1]
    nominal_coverage = hi_q - lo_q
    alpha = 1 - nominal_coverage
    lower_col, upper_col = f"forecast_q{int(lo_q*100)}", f"forecast_q{int(hi_q*100)}"

    pooled_calib = evaluate_interval_calibration(
        merged[target_col], merged[lower_col], merged[upper_col], nominal_coverage
    )
    by_fold_calib = (
        evaluate_interval_calibration_by_fold(merged, target_col, lower_col, upper_col, nominal_coverage, fold_col)
        if fold_col in merged.columns else None
    )
    width = merged[upper_col] - merged[lower_col]
    interval_score = compute_interval_score(merged[target_col], merged[lower_col], merged[upper_col], alpha)

    result = {
        "window_days": window_days,
        "min_periods_days": min_periods_days,
        "pooled_calibration": pooled_calib,
        "by_fold_calibration": by_fold_calib,
        "mean_width": float(width.mean()),
        "median_width": float(width.median()),
        "interval_score": interval_score,
        "n_warmup": int(merged[lower_col].isna().sum()),
    }
    if return_per_row:
        per_row_cols = [ts_col, "prediction", lower_col, upper_col]
        if fold_col in merged.columns:
            per_row_cols.append(fold_col)
        result["per_row"] = merged[per_row_cols].copy()
    return result


def compute_rolling_coverage_diagnostics(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    ts_col_values: pd.Series,
    window_days: int = 30,
) -> pd.DataFrame:
    """DIAGNOSTIC ONLY -- never used for window selection. A pooled or
    per-fold coverage number can conceal a period of genuinely bad
    calibration: a method could show 80% coverage over three months
    while experiencing one 30-day stretch at 60%, with over-coverage
    elsewhere in the period masking it in every aggregate reported so
    far. This computes trailing (not centered -- point-in-time safe,
    same closed='left' discipline as the rest of this module) 30-day
    coverage, below/above fractions, and mean interval width, so that
    stretch is directly visible rather than averaged away.

    Explicitly NOT used as a selection criterion anywhere in this
    project: it answers "how bad can a 30-day stretch get," not "which
    candidate is best on average" -- the latter is what interval_score
    already exists for.
    """
    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(ts_col_values, utc=True),
        "actual": actual.values, "lower": lower.values, "upper": upper.values,
    }).sort_values("timestamp_utc").reset_index(drop=True)

    within = ((df["actual"] >= df["lower"]) & (df["actual"] <= df["upper"])).astype(float)
    below = (df["actual"] < df["lower"]).astype(float)
    above = (df["actual"] > df["upper"]).astype(float)
    width = df["upper"] - df["lower"]

    indexed = df.set_index("timestamp_utc")
    # closed='right' (INCLUDING the current row) here, unlike the
    # leakage-safety rolling elsewhere in this module -- this is a
    # retrospective diagnostic ("how did the last 30 days actually go"),
    # not a forecast input, so there is no leakage concern: it never
    # feeds back into any interval computation.
    roll_within = pd.Series(within.values, index=indexed.index).rolling(f"{window_days}D", closed="right")
    roll_below = pd.Series(below.values, index=indexed.index).rolling(f"{window_days}D", closed="right")
    roll_above = pd.Series(above.values, index=indexed.index).rolling(f"{window_days}D", closed="right")
    roll_width = pd.Series(width.values, index=indexed.index).rolling(f"{window_days}D", closed="right")

    return pd.DataFrame({
        "timestamp_utc": df["timestamp_utc"],
        "rolling_coverage": roll_within.mean().values,
        "rolling_frac_below": roll_below.mean().values,
        "rolling_frac_above": roll_above.mean().values,
        "rolling_mean_width": roll_width.mean().values,
        "rolling_n": roll_within.count().values,
    })


def summarize_worst_rolling_window(rolling_diagnostics: pd.DataFrame, nominal_coverage: float) -> dict:
    """Extracts the single worst rolling-window observation -- the
    number a risk manager actually cares about ("how bad can it get"),
    not the average. Ignores rows with fewer than half the intended
    window's worth of observations (the diagnostic's own warm-up), so
    an artificially extreme reading from 2 data points at the very
    start doesn't masquerade as the worst real stretch.
    """
    valid = rolling_diagnostics[
        rolling_diagnostics["rolling_n"] >= rolling_diagnostics["rolling_n"].max() / 2
    ]
    if valid.empty:
        return {"worst_coverage": None, "worst_coverage_timestamp": None}
    worst_idx = valid["rolling_coverage"].idxmin()
    worst_row = valid.loc[worst_idx]
    return {
        "worst_coverage": float(worst_row["rolling_coverage"]),
        "worst_coverage_timestamp": str(worst_row["timestamp_utc"]),
        "worst_coverage_shortfall_from_nominal": float(nominal_coverage - worst_row["rolling_coverage"]),
        "frac_below_at_worst": float(worst_row["rolling_frac_below"]),
        "frac_above_at_worst": float(worst_row["rolling_frac_above"]),
    }


def compute_residual_quantile_offset(
    residual_series: pd.DataFrame,
    quantile: float,
    window_days: int,
    min_periods_days: int,
    ts_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """Returns just the ADDITIVE OFFSET for one quantile (forecast_qXX
    minus prediction), not the combined point+offset forecast --
    needed specifically to recombine ONE model's point forecast with a
    DIFFERENT model's uncertainty envelope (e.g. "Full's point forecast
    with Tier-1's residual-quantile width"), which is not expressible
    from compute_rolling_residual_quantiles() alone since that always
    adds a series's own offset to its own prediction.

    Same leakage-safety guarantee as compute_rolling_residual_quantiles
    (closed='left', offset at t built only from residuals strictly
    before t) -- this is a thin wrapper that subtracts prediction back
    out, not a separate computation path that could drift out of sync.
    """
    qf = compute_rolling_residual_quantiles(residual_series, [quantile], window_days, min_periods_days, ts_col)
    q_col = f"forecast_q{int(quantile*100)}"
    return pd.DataFrame({
        ts_col: qf[ts_col],
        "offset": qf[q_col] - qf["prediction"],
    })
