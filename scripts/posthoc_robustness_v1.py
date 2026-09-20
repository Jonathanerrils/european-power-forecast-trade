"""Post-hoc robustness diagnostics for the exposed Jan-Jul 2026 holdout.

IMPORTANT: These analyses were introduced after holdout exposure. They are
supplementary diagnostics and do not redefine the frozen confirmation rule.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "outputs/holdout/delu_features/holdout_v1/hourly_holdout_inputs.csv"
D = ROOT / "outputs/holdout/delu_features/holdout_v1/per_day_results.csv"
OUT = ROOT / "outputs/posthoc/robustness_v1"
OUT.mkdir(parents=True, exist_ok=True)

ETA = 0.85
C_PARAM = 10.0
TOTAL_COST = C_PARAM * (1.0 + ETA)
ALPHA = 0.20
SEED = 20260920
BOOTSTRAP_REPS = 20_000
BLOCK = 7

h = pd.read_csv(H, parse_dates=["timestamp_utc"])
h = h.loc[h["common_evaluation_day"].astype(bool)].copy()
h["delivery_date"] = pd.to_datetime(h["delivery_date"])
h["local_hour"] = h["timestamp_utc"].dt.tz_convert("Europe/Berlin").dt.hour

d = pd.read_csv(D, parse_dates=["delivery_date"]).sort_values("delivery_date")
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
        return 0.0
    _, i, j = best
    return ETA * actual[j] - actual[i] - TOTAL_COST

# Weekday-aware naive: lag-24 Tue-Fri, lag-168 Sat-Mon.
bench = []
grouped = {k: v.sort_values("timestamp_utc").copy()
           for k, v in h.groupby("delivery_date", sort=True)}
dates = sorted(grouped)
for date in dates:
    day = grouped[date]
    use24 = date.dayofweek in (1, 2, 3, 4)  # Tue-Fri, Monday=0
    pred = day["lag_24_pred"] if use24 else day["lag_168_pred"]
    bench.append((date, "weekday_naive", choose_trade(day, pred)))

# Trailing seven completed delivery-day mean profile; begins on day 8.
for ix in range(7, len(dates)):
    date = dates[ix]
    day = grouped[date]
    hist_dates = dates[ix-7:ix]
    preds = []
    for hour in day["local_hour"]:
        vals = []
        for hd in hist_dates:
            vals.extend(grouped[hd].loc[grouped[hd]["local_hour"].eq(hour),
                                         "price_eur_mwh"].tolist())
        preds.append(float(np.mean(vals)))
    bench.append((date, "trailing7_profile", choose_trade(day, preds)))

bench = pd.DataFrame(bench, columns=["delivery_date", "strategy", "net_pnl"])
bench.to_csv(OUT / "posthoc_benchmark_per_day.csv", index=False)

# Common 205-day comparison from holdout day 8.
common_dates = dates[7:]
rows = []
for name, pred_col in [
    ("Lag-24", "lag_24_pred"),
    ("XGBoost Full", "xgboost_full_pred"),
    ("XGBoost Tier-1", "xgboost_tier1_pred"),
]:
    sub = h[h["delivery_date"].isin(common_dates)]
    err = sub["price_eur_mwh"] - sub[pred_col]
    pnl_col = {"Lag-24":"S1_net_pnl","XGBoost Full":"S2_net_pnl",
               "XGBoost Tier-1":"S4_net_pnl"}[name]
    pnl = dmap.loc[common_dates, pnl_col].sum()
    oracle = dmap.loc[common_dates, "oracle_pnl"].sum()
    rows.append([name, len(sub), np.abs(err).mean(),
                 np.sqrt(np.mean(err**2)), pnl, pnl/oracle])

# Weekday-aware forecast metrics.
sub = h[h["delivery_date"].isin(common_dates)].copy()
use24 = sub["delivery_date"].dt.dayofweek.isin([1,2,3,4])
pred = np.where(use24, sub["lag_24_pred"], sub["lag_168_pred"])
err = sub["price_eur_mwh"].to_numpy() - pred
wpnl = bench.query("strategy == 'weekday_naive' and delivery_date in @common_dates")["net_pnl"].sum()
oracle = dmap.loc[common_dates, "oracle_pnl"].sum()
rows.append(["Weekday-aware naive", len(sub), np.abs(err).mean(),
             np.sqrt(np.mean(err**2)), wpnl, wpnl/oracle])

# Trailing-7 forecast metrics.
err_abs, err_sq = [], []
for ix in range(7, len(dates)):
    date = dates[ix]
    day = grouped[date]
    hist_dates = dates[ix-7:ix]
    for _, r in day.iterrows():
        vals=[]
        for hd in hist_dates:
            vals.extend(grouped[hd].loc[grouped[hd]["local_hour"].eq(r["local_hour"]),
                                         "price_eur_mwh"].tolist())
        p=float(np.mean(vals)); e=float(r["price_eur_mwh"])-p
        err_abs.append(abs(e)); err_sq.append(e*e)
tpnl = bench.query("strategy == 'trailing7_profile'")["net_pnl"].sum()
rows.append(["Trailing 7-day mean profile", len(err_abs), np.mean(err_abs),
             np.sqrt(np.mean(err_sq)), tpnl, tpnl/oracle])

pd.DataFrame(rows, columns=["strategy","n_hours","mae","rmse","net_pnl","oracle_share"]).to_csv(
    OUT / "posthoc_benchmark_summary.csv", index=False)

def moving_block_ci(values, reps=BOOTSTRAP_REPS, block=BLOCK, seed=SEED):
    x = np.asarray(values, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    totals = np.empty(reps)
    for b in range(reps):
        out=[]
        while len(out)<n:
            start=int(rng.integers(0,n))
            out.extend(x[(start + np.arange(block)) % n])
        totals[b]=np.sum(out[:n])
    return np.quantile(totals,[.025,.975])

contrasts = {
    "S2-S1": d["S2_net_pnl"]-d["S1_net_pnl"],
    "S3-S2": d["S3_net_pnl"]-d["S2_net_pnl"],
    "S4-S2": d["S4_net_pnl"]-d["S2_net_pnl"],
    "S5-S3": d["S5_net_pnl"]-d["S3_net_pnl"],
}
inf=[]
for name,x in contrasts.items():
    lo,hi=moving_block_ci(x)
    inf.append([name,len(x),x.sum(),x.mean(),x.std(ddof=1),
                lo,hi,(x>0).sum(),(x<0).sum(),(x==0).sum()])

# Post-hoc S2/S4 versus trailing profile on common 205 days.
tr = bench.query("strategy == 'trailing7_profile'").set_index("delivery_date")["net_pnl"]
for name,col in [("S2-trailing7","S2_net_pnl"),("S4-trailing7","S4_net_pnl")]:
    x=dmap.loc[common_dates,col]-tr.loc[common_dates]
    lo,hi=moving_block_ci(x)
    inf.append([name,len(x),x.sum(),x.mean(),x.std(ddof=1),
                lo,hi,(x>0).sum(),(x<0).sum(),(x==0).sum()])

pd.DataFrame(inf, columns=["contrast","n_days","total_difference","mean_daily_difference",
                           "sd_daily_difference","block7_ci_low","block7_ci_high",
                           "wins","losses","ties"]).to_csv(
    OUT / "posthoc_block_bootstrap.csv", index=False)

m = d.assign(month=d["delivery_date"].dt.to_period("M").astype(str),
             delta=d["S2_net_pnl"]-d["S1_net_pnl"])
m.groupby("month").agg(n_days=("delta","size"),s2_minus_s1=("delta","sum")).reset_index().to_csv(
    OUT / "posthoc_monthly_s2_s1.csv", index=False)

def calibration(frame, lo, hi):
    y=frame["price_eur_mwh"].to_numpy(float)
    L=frame[lo].to_numpy(float); U=frame[hi].to_numpy(float)
    lower=y<L; upper=y>U; covered=~(lower|upper)
    score=(U-L)+(2/ALPHA)*(L-y)*lower+(2/ALPHA)*(y-U)*upper
    return [len(frame),covered.mean(),lower.mean(),upper.mean(),
            np.mean(U-L),np.mean(score)]

overall=[]
for label,lo,hi in [("Full","full_L","full_U"),("Tier-1","tier1_L","tier1_U")]:
    overall.append([label]+calibration(h,lo,hi))
pd.DataFrame(overall,columns=["information_set","n","coverage","lower_miss","upper_miss",
                              "mean_width","winkler_score"]).to_csv(
    OUT/"posthoc_calibration_overall.csv",index=False)

month_rows=[]
h["month"]=h["delivery_date"].dt.to_period("M").astype(str)
for month,g in h.groupby("month"):
    for label,lo,hi in [("Full","full_L","full_U"),("Tier-1","tier1_L","tier1_U")]:
        month_rows.append([month,label]+calibration(g,lo,hi))
pd.DataFrame(month_rows,columns=["month","information_set","n","coverage","lower_miss","upper_miss",
                                 "mean_width","winkler_score"]).to_csv(
    OUT/"posthoc_calibration_monthly.csv",index=False)

hour_rows=[]
for hour,g in h.groupby("local_hour"):
    for label,lo,hi in [("Full","full_L","full_U"),("Tier-1","tier1_L","tier1_U")]:
        hour_rows.append([hour,label]+calibration(g,lo,hi))
pd.DataFrame(hour_rows,columns=["local_hour","information_set","n","coverage","lower_miss","upper_miss",
                                "mean_width","winkler_score"]).to_csv(
    OUT/"posthoc_calibration_by_hour.csv",index=False)
