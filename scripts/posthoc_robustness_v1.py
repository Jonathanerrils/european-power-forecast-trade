"""Post-hoc robustness diagnostics for the exposed Jan-Jul 2026 holdout.

IMPORTANT: These analyses were introduced after holdout exposure. They are
supplementary diagnostics and do not redefine the frozen sign-consistency rule.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "outputs/holdout/delu_features/holdout_v1/hourly_holdout_inputs.csv"
D = ROOT / "outputs/holdout/delu_features/holdout_v1/per_day_results.csv"
DEV_HIST = ROOT / "outputs/models/delu_features/xgboost_v1_a03fix/regime_stress_test_predictions.csv"
OUT = ROOT / "outputs/posthoc/robustness_v1"
OUT.mkdir(parents=True, exist_ok=True)

ETA = 0.85
C_PARAM = 10.0
TOTAL_COST = C_PARAM * (1.0 + ETA)
ALPHA = 0.20
SEED = 20260920
BOOTSTRAP_REPS = 20_000
BLOCK = 7
PROFILE_WINDOWS = [3, 7, 10, 14, 21, 28]

h = pd.read_csv(H, parse_dates=["timestamp_utc"])
h = h.loc[h["common_evaluation_day"].astype(bool)].copy()
h["delivery_date"] = pd.to_datetime(h["delivery_date"]).dt.normalize()
h["local_hour"] = h["timestamp_utc"].dt.tz_convert("Europe/Berlin").dt.hour

dev = pd.read_csv(DEV_HIST, parse_dates=["timestamp_utc"])
dev["delivery_date"] = (
    dev["timestamp_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None).dt.normalize()
)
dev["local_hour"] = dev["timestamp_utc"].dt.tz_convert("Europe/Berlin").dt.hour

d = pd.read_csv(D, parse_dates=["delivery_date"]).sort_values("delivery_date")
d["delivery_date"] = d["delivery_date"].dt.normalize()
dmap = d.set_index("delivery_date")

def choose_trade(day, pred):
    pred = np.asarray(pred, float)
    actual = day["price_eur_mwh"].to_numpy(float)
    best = (-np.inf, None, None)
    for i in range(len(day)):
        for j in range(i + 1, len(day)):
            score = ETA * pred[j] - pred[i] - TOTAL_COST
            if score > best[0]:
                best = (score, i, j)
    if best[0] <= 0:
        return {"traded": False, "score": best[0], "net_pnl": 0.0, "i": None, "j": None}
    score, i, j = best
    return {
        "traded": True,
        "score": score,
        "net_pnl": ETA * actual[j] - actual[i] - TOTAL_COST,
        "i": i,
        "j": j,
    }

grouped = {
    k: v.sort_values("timestamp_utc").copy()
    for k, v in h.groupby("delivery_date", sort=True)
}
dates = sorted(grouped)

# Historical local-delivery-date price vectors used only by the post-hoc
# trailing-profile benchmark. The Oct-Dec 2025 regime-stress prediction artifact
# supplies the pre-holdout price history needed to avoid an artificial Jan 1-7
# burn-in. By D-1 11:45, the day-ahead price vector for delivery date D-1 has
# already been published, so these prior local-delivery-date price vectors are
# admissible inputs to this descriptive benchmark.
history = {
    k: v[["local_hour", "price_eur_mwh"]].copy()
    for k, v in dev.groupby("delivery_date", sort=True)
}
for k, v in grouped.items():
    history[k] = v[["local_hour", "price_eur_mwh"]].copy()
history_dates = sorted(history)

def profile_forecast(date, window):
    prior = [x for x in history_dates if x < date][-window:]
    if len(prior) != window:
        raise RuntimeError(f"Need {window} prior price profiles before {date.date()}; found {len(prior)}")
    day = grouped[date]
    preds = []
    for hour in day["local_hour"]:
        vals = []
        for hd in prior:
            vals.extend(
                history[hd].loc[history[hd]["local_hour"].eq(hour), "price_eur_mwh"].tolist()
            )
        if not vals:
            raise RuntimeError(f"No historical price for local hour {hour} before {date.date()}")
        preds.append(float(np.mean(vals)))
    return np.asarray(preds, float)

# Weekday-aware naive and trailing-profile benchmark family.
bench = []
forecast_cache = {}
for date in dates:
    day = grouped[date]
    use24 = date.dayofweek in (1, 2, 3, 4)  # Tue-Fri, Monday=0
    weekday_pred = day["lag_24_pred"].to_numpy(float) if use24 else day["lag_168_pred"].to_numpy(float)
    res = choose_trade(day, weekday_pred)
    bench.append((date, "weekday_naive", res["net_pnl"]))
    forecast_cache[(date, "weekday_naive")] = weekday_pred

    for window in PROFILE_WINDOWS:
        pred = profile_forecast(date, window)
        res = choose_trade(day, pred)
        bench.append((date, f"trailing{window}_profile", res["net_pnl"]))
        forecast_cache[(date, f"trailing{window}_profile")] = pred

bench = pd.DataFrame(bench, columns=["delivery_date", "strategy", "net_pnl"])
bench.to_csv(OUT / "posthoc_benchmark_per_day.csv", index=False)

# Full 212-day/5,087-hour benchmark comparison.
oracle = d["oracle_pnl"].sum()
rows = []
for name, pred_col, pnl_col in [
    ("Lag-24", "lag_24_pred", "S1_net_pnl"),
    ("XGBoost Full", "xgboost_full_pred", "S2_net_pnl"),
    ("XGBoost Tier-1", "xgboost_tier1_pred", "S4_net_pnl"),
]:
    err = h["price_eur_mwh"] - h[pred_col]
    pnl = d[pnl_col].sum()
    rows.append([name, len(h), np.abs(err).mean(), np.sqrt(np.mean(err**2)), pnl, pnl / oracle])

for strategy, label in [
    ("weekday_naive", "Weekday-aware naive"),
    ("trailing7_profile", "Trailing 7-day mean profile"),
]:
    err_abs, err_sq = [], []
    for date in dates:
        day = grouped[date]
        pred = forecast_cache[(date, strategy)]
        err = day["price_eur_mwh"].to_numpy(float) - pred
        err_abs.extend(np.abs(err))
        err_sq.extend(err**2)
    pnl = bench.loc[bench["strategy"].eq(strategy), "net_pnl"].sum()
    rows.append([label, len(err_abs), np.mean(err_abs), np.sqrt(np.mean(err_sq)), pnl, pnl / oracle])

summary = pd.DataFrame(
    rows, columns=["strategy", "n_hours", "mae", "rmse", "net_pnl", "oracle_share"]
)
order = ["Lag-24", "Weekday-aware naive", "Trailing 7-day mean profile", "XGBoost Tier-1", "XGBoost Full"]
summary["strategy"] = pd.Categorical(summary["strategy"], categories=order, ordered=True)
summary.sort_values("strategy").to_csv(OUT / "posthoc_benchmark_summary.csv", index=False)

# Profile-window sensitivity, introduced after holdout exposure.
full_pnl = float(d["S2_net_pnl"].sum())
lag24_pnl = float(d["S1_net_pnl"].sum())
full_gain = full_pnl - lag24_pnl
window_rows = []
for window in PROFILE_WINDOWS:
    strategy = f"trailing{window}_profile"
    pnl = float(bench.loc[bench["strategy"].eq(strategy), "net_pnl"].sum())
    gain = pnl - lag24_pnl
    window_rows.append([window, pnl, full_pnl - pnl, gain, gain / full_gain])
pd.DataFrame(
    window_rows,
    columns=["window_days", "benchmark_net_pnl", "full_minus_benchmark",
             "benchmark_gain_over_lag24", "share_of_full_gain_over_lag24"],
).to_csv(OUT / "posthoc_profile_window_sensitivity.csv", index=False)

# Deterministic xorshift32 RNG so archived bootstrap CSVs are exactly reproducible.
def _xorshift32(seed):
    state = int(seed) & 0xFFFFFFFF
    while True:
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= (state >> 17) & 0xFFFFFFFF
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        yield state / 2**32

def moving_block_ci(values, reps=BOOTSTRAP_REPS, block=BLOCK, seed=SEED):
    x = np.asarray(values, float)
    n = len(x)
    rng = _xorshift32(seed)
    totals = np.empty(reps)
    for b in range(reps):
        total = 0.0
        left = n
        while left:
            start = int(next(rng) * n)
            take = min(block, left)
            for k in range(take):
                total += x[(start + k) % n]
            left -= take
        totals[b] = total
    return np.quantile(totals, [0.025, 0.975])

contrasts = {
    "S2-S1": d["S2_net_pnl"] - d["S1_net_pnl"],
    "S3-S2": d["S3_net_pnl"] - d["S2_net_pnl"],
    "S4-S2": d["S4_net_pnl"] - d["S2_net_pnl"],
    "S5-S3": d["S5_net_pnl"] - d["S3_net_pnl"],
}
tr7 = bench.query("strategy == 'trailing7_profile'").set_index("delivery_date")["net_pnl"]
contrasts["S2-trailing7"] = dmap.loc[dates, "S2_net_pnl"] - tr7.loc[dates]
contrasts["S4-trailing7"] = dmap.loc[dates, "S4_net_pnl"] - tr7.loc[dates]

inf = []
for name, x in contrasts.items():
    x = pd.Series(x, dtype=float)
    lo, hi = moving_block_ci(x)
    inf.append([
        name, len(x), x.sum(), x.mean(), x.std(ddof=1),
        lo, hi, (x > 0).sum(), (x < 0).sum(), (x == 0).sum()
    ])
pd.DataFrame(
    inf,
    columns=["contrast", "n_days", "total_difference", "mean_daily_difference",
             "sd_daily_difference", "block7_ci_low", "block7_ci_high",
             "wins", "losses", "ties"],
).to_csv(OUT / "posthoc_block_bootstrap.csv", index=False)

m = d.assign(
    month=d["delivery_date"].dt.to_period("M").astype(str),
    delta=d["S2_net_pnl"] - d["S1_net_pnl"],
)
m.groupby("month").agg(n_days=("delta", "size"), s2_minus_s1=("delta", "sum")).reset_index().to_csv(
    OUT / "posthoc_monthly_s2_s1.csv", index=False
)

# Fixed-threshold frontier for interpreting the pooled-residual S3 rule.
frontier_rows = []
s3_trade = dmap.loc[dates, "S3_traded"].map(lambda x: str(x).lower() == "true")
for tau in [0, 20, 30, 40, 42.5, 50, 60]:
    daily_pnl = []
    traded = []
    for date in dates:
        day = grouped[date]
        res = choose_trade(day, day["xgboost_full_pred"].to_numpy(float))
        take = res["score"] > tau
        traded.append(take)
        daily_pnl.append(res["net_pnl"] if take else 0.0)
    daily_pnl = np.asarray(daily_pnl, float)
    traded = np.asarray(traded, bool)
    ntrades = int(traded.sum())
    hit = float((daily_pnl[traded] > 0).mean()) if ntrades else np.nan
    cum = np.cumsum(daily_pnl)
    maxdd = float(np.min(cum - np.maximum.accumulate(cum)))
    overlap = int(np.sum(traded & s3_trade.to_numpy(bool)))
    frontier_rows.append([
        tau, ntrades, daily_pnl.sum(), hit, daily_pnl.min(), maxdd,
        overlap, int(np.sum(traded & ~s3_trade.to_numpy(bool))),
        int(np.sum(~traded & s3_trade.to_numpy(bool))),
    ])
pd.DataFrame(
    frontier_rows,
    columns=["threshold", "trades", "net_pnl", "hit_rate", "worst_day", "max_drawdown",
             "overlap_with_s3", "threshold_only", "s3_only"],
).to_csv(OUT / "posthoc_threshold_frontier.csv", index=False)

def calibration(frame, lo, hi):
    y = frame["price_eur_mwh"].to_numpy(float)
    L = frame[lo].to_numpy(float)
    U = frame[hi].to_numpy(float)
    lower = y < L
    upper = y > U
    covered = ~(lower | upper)
    score = (U-L) + (2/ALPHA)*(L-y)*lower + (2/ALPHA)*(y-U)*upper
    return [len(frame), covered.mean(), lower.mean(), upper.mean(), np.mean(U-L), np.mean(score)]

overall = []
for label, lo, hi in [("Full", "full_L", "full_U"), ("Tier-1", "tier1_L", "tier1_U")]:
    overall.append([label] + calibration(h, lo, hi))
pd.DataFrame(
    overall,
    columns=["information_set", "n", "coverage", "lower_miss", "upper_miss",
             "mean_width", "winkler_score"],
).to_csv(OUT / "posthoc_calibration_overall.csv", index=False)

month_rows = []
h["month"] = h["delivery_date"].dt.to_period("M").astype(str)
for month, g in h.groupby("month"):
    for label, lo, hi in [("Full", "full_L", "full_U"), ("Tier-1", "tier1_L", "tier1_U")]:
        month_rows.append([month, label] + calibration(g, lo, hi))
pd.DataFrame(
    month_rows,
    columns=["month", "information_set", "n", "coverage", "lower_miss", "upper_miss",
             "mean_width", "winkler_score"],
).to_csv(OUT / "posthoc_calibration_monthly.csv", index=False)

hour_rows = []
for hour, g in h.groupby("local_hour"):
    for label, lo, hi in [("Full", "full_L", "full_U"), ("Tier-1", "tier1_L", "tier1_U")]:
        hour_rows.append([hour, label] + calibration(g, lo, hi))
pd.DataFrame(
    hour_rows,
    columns=["local_hour", "information_set", "n", "coverage", "lower_miss", "upper_miss",
             "mean_width", "winkler_score"],
).to_csv(OUT / "posthoc_calibration_by_hour.csv", index=False)
