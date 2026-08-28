"""Run locally: python run_eda.py [input_filename]

EDA on the frozen feature matrix (default: data/processed/delu_features.parquet),
restricted to the DEVELOPMENT sample only (config.yaml's splits.holdout_start
onward is excluded). Per spec section 9, 2026+ is the final untouched
holdout -- even descriptive EDA on it risks unconsciously shaping later
modelling choices, so it's excluded by default here, not just at
train/test-split time. Use --include-holdout to override for a
deliberate, separate post-freeze diagnostic pass only.

Per spec section 6 (understand prices, negative-price periods, spikes,
seasonality, load, renewables, residual load -- don't tune models yet)
and section 25 (a handful of high-value figures, not twenty decorative
ones). At this stage (before any baseline/model exists), the figures
that matter are the ones about the TARGET and the FUNDAMENTALS
relationship -- forecast-vs-actual and P&L figures come later once
there's a model/strategy to compare against.

Prints numeric summaries to console (so they can be shared/verified as
text) and saves PNGs to outputs/figures/ (so they can be inspected
visually or uploaded for review).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import pandas as pd

from src.clean import local_delivery_date_to_utc
from src.utils import load_config, setup_logging, REPO_ROOT

REQUIRED_COLS = {
    "timestamp_utc", "delivery_date", "price_eur_mwh", "load_forecast_mw",
    "wind_onshore_forecast_mw", "wind_offshore_forecast_mw", "solar_forecast_mw",
    "renewables_forecast_mw", "residual_load_forecast_mw", "renewable_share_forecast",
    "post_15min_mtu", "hour_local", "dow_local", "month_local", "weekend",
}


def resolve_run_args(args: list) -> tuple:
    """input_filename and run_version, plus the --include-holdout flag
    parsed separately by the caller. run_version is MANDATORY, not
    defaulted -- an earlier version of this script used a fixed,
    unversioned output path (outputs/eda/<stem>/<scope>/), so every
    re-run silently overwrote the previous EDA output with no guard and
    no history. Concretely: after the A03 parser fix, re-running EDA
    on the corrected data overwrote the ONLY copy of the pre-fix EDA
    output, making a real pre/post comparison impossible after the
    fact -- exactly the kind of loss run_models.py's mandatory,
    FileExistsError-guarded run_version was already designed to
    prevent elsewhere in this project. EDA gets the same treatment now.
    """
    if len(args) not in (1, 2):
        raise SystemExit(
            "Usage:\n"
            "  python run_eda.py <input_filename> <run_version> [--include-holdout]\n\n"
            "Example:\n"
            "  python run_eda.py delu_features.parquet development_a03fix_v1"
        )
    if len(args) == 1:
        raise SystemExit(
            "run_version is now mandatory (previously this script silently overwrote "
            "the same fixed output path on every run -- see resolve_run_args's docstring "
            "for why that was a real problem, not just a style preference).\n\n"
            "Usage:\n"
            "  python run_eda.py <input_filename> <run_version> [--include-holdout]"
        )
    return args[0], args[1]


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    include_holdout = "--include-holdout" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    input_filename, run_version = resolve_run_args(args)

    in_path = REPO_ROOT / cfg["data"]["processed_dir"] / input_filename
    if not in_path.exists():
        available = list((REPO_ROOT / cfg["data"]["processed_dir"]).glob("*.parquet"))
        raise FileNotFoundError(
            f"{in_path} not found. Run run_features.py first, or pass the "
            f"correct filename: python run_eda.py <filename>.\n"
            f"Files found: {[p.name for p in available] or 'none'}"
        )

    df = pd.read_parquet(in_path)

    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"EDA input is missing required columns: {sorted(missing_cols)}. "
            f"These are expected to be part of the feature-matrix contract "
            f"(run_features.py) -- fix upstream rather than silently skip them here."
        )

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Scope outputs by dataset + holdout-inclusion + run_version, so a
    # post-freeze --include-holdout run can never silently overwrite the
    # clean development-only evidence, AND so re-running EDA (e.g. after
    # a data-correction like the A03 fix) preserves the previous run for
    # comparison instead of silently destroying it.
    scope_name = "all_postfreeze" if include_holdout else "development"
    base_dir = REPO_ROOT / "outputs" / "eda" / in_path.stem / scope_name / run_version
    if base_dir.exists() and any(base_dir.iterdir()):
        raise FileExistsError(
            f"{base_dir} already contains results. EDA runs are meant to be preserved "
            f"for comparison -- pass a new run_version instead of overwriting it, e.g.\n"
            f"  python run_eda.py {input_filename} {run_version}_v2"
        )
    fig_dir = base_dir / "figures"
    table_dir = base_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Holdout exclusion -- do this before anything else touches the data
    # -----------------------------------------------------------------
    n_total = len(df)
    holdout_excluded_rows = 0
    if not include_holdout:
        holdout_start = local_delivery_date_to_utc(
            cfg["splits"]["holdout_start"], local_tz=cfg["market"]["timezone_local"]
        )
        holdout_mask = df["timestamp_utc"] >= holdout_start
        holdout_excluded_rows = int(holdout_mask.sum())
        print(
            f"EDA scope: development sample only. Excluding {holdout_excluded_rows:,} "
            f"holdout row(s) from {cfg['splits']['holdout_start']} onward "
            f"(pass --include-holdout to override for a deliberate post-freeze pass)."
        )
        df = df.loc[~holdout_mask].reset_index(drop=True)
    else:
        print("WARNING: --include-holdout set. This run includes the final holdout "
              "together with the development sample. Only use after model freeze.")

    if df.empty:
        raise ValueError("EDA development sample is empty after holdout exclusion.")
    if df["price_eur_mwh"].notna().sum() == 0:
        raise ValueError("EDA development sample contains no usable price observations.")

    price = df["price_eur_mwh"]

    # -----------------------------------------------------------------
    # 1. Price distribution summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRICE DISTRIBUTION SUMMARY (EUR/MWh)")
    print("=" * 70)
    desc = price.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(desc.to_string())
    n_negative = int((price < 0).sum())
    n_zero = int((price == 0).sum())
    n_above_200 = int((price > 200).sum())
    n_above_500 = int((price > 500).sum())
    print(f"\nNegative prices:   {n_negative} ({n_negative / price.notna().sum():.2%})")
    print(f"Zero prices:       {n_zero} ({n_zero / price.notna().sum():.2%})")
    print(f"Prices > 200:      {n_above_200} ({n_above_200 / price.notna().sum():.2%})")
    print(f"Prices > 500:      {n_above_500} ({n_above_500 / price.notna().sum():.2%})")

    print("\nTop 10 highest-price hours:")
    print(df.nlargest(10, "price_eur_mwh")[["timestamp_utc", "price_eur_mwh"]].to_string(index=False))
    print("\nTop 10 lowest-price hours:")
    print(df.nsmallest(10, "price_eur_mwh")[["timestamp_utc", "price_eur_mwh"]].to_string(index=False))

    desc.to_csv(table_dir / "price_distribution_summary.csv")

    # -----------------------------------------------------------------
    # 2. Fundamentals summary (before jumping to residual load vs price)
    # -----------------------------------------------------------------
    fundamental_cols = [
        "load_forecast_mw", "wind_onshore_forecast_mw", "wind_offshore_forecast_mw",
        "solar_forecast_mw", "renewables_forecast_mw", "residual_load_forecast_mw",
        "renewable_share_forecast",
    ]  # all required by REQUIRED_COLS -- no need to filter against df.columns

    fundamentals_summary = df[fundamental_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    print("\n" + "=" * 70)
    print("FUNDAMENTALS SUMMARY")
    print("=" * 70)
    print(fundamentals_summary.to_string())
    fundamentals_summary.to_csv(table_dir / "fundamentals_summary.csv")

    fundamentals_missingness = pd.DataFrame({
        "missing_n": df[fundamental_cols].isna().sum(),
        "missing_pct": df[fundamental_cols].isna().mean() * 100,
    })
    print("\nFundamental-feature missingness:")
    print(fundamentals_missingness.to_string())
    fundamentals_missingness.to_csv(table_dir / "fundamentals_missingness.csv")

    n_negative_residual = int((df["residual_load_forecast_mw"] < 0).sum()) if "residual_load_forecast_mw" in df.columns else None
    if n_negative_residual is not None:
        print(f"\nHours with negative residual load forecast "
              f"(forecast renewables > forecast load): {n_negative_residual} "
              f"({n_negative_residual / df['residual_load_forecast_mw'].notna().sum():.2%})")

    # -----------------------------------------------------------------
    # 3. Seasonality: by hour, by day of week, by month, by year
    # -----------------------------------------------------------------
    by_hour = df.groupby("hour_local")["price_eur_mwh"].agg(["mean", "std", "count"])
    by_dow = df.groupby("dow_local")["price_eur_mwh"].agg(["mean", "std", "count"])
    by_month = df.groupby("month_local")["price_eur_mwh"].agg(["mean", "std", "count"])
    df["year_local"] = pd.to_datetime(df["delivery_date"]).dt.year
    by_year = df.groupby("year_local")["price_eur_mwh"].agg(["mean", "median", "std", "min", "max", "count"])

    print("\n" + "=" * 70)
    print("SEASONALITY")
    print("=" * 70)
    print("\nMean price by local hour of day:")
    print(by_hour.to_string())
    print("\nMean price by day of week (0=Mon):")
    print(by_dow.to_string())
    print("\nMean/median price by year:")
    print(by_year.to_string())

    by_hour.to_csv(table_dir / "price_by_hour.csv")
    by_dow.to_csv(table_dir / "price_by_dow.csv")
    by_month.to_csv(table_dir / "price_by_month.csv")
    by_year.to_csv(table_dir / "price_by_year.csv")

    # -----------------------------------------------------------------
    # 4. Residual load vs price relationship (Pearson + Spearman, overall and by year)
    # -----------------------------------------------------------------
    valid = df[["residual_load_forecast_mw", "price_eur_mwh", "year_local"]].dropna(
        subset=["residual_load_forecast_mw", "price_eur_mwh"]
    )
    pearson_corr = valid["residual_load_forecast_mw"].corr(valid["price_eur_mwh"], method="pearson")
    spearman_corr = valid["residual_load_forecast_mw"].corr(valid["price_eur_mwh"], method="spearman")

    print("\n" + "=" * 70)
    print("RESIDUAL LOAD vs PRICE")
    print("=" * 70)
    print(f"Pearson correlation:  {pearson_corr:.3f}")
    print(f"Spearman correlation: {spearman_corr:.3f}")
    print(f"(n = {len(valid)} rows with both values present)")
    print("\nA single overall coefficient can be dominated by extreme-price periods "
          "(e.g. 2022). Breaking out by year shows whether the relationship is a "
          "stable structural driver or regime-dependent:")

    corr_by_year = (
        valid.groupby("year_local")
        .apply(
            lambda x: pd.Series({
                "pearson": x["residual_load_forecast_mw"].corr(x["price_eur_mwh"], method="pearson"),
                "spearman": x["residual_load_forecast_mw"].corr(x["price_eur_mwh"], method="spearman"),
                "n": len(x),
            }),
            include_groups=False,
        )
    )
    print("\nResidual load vs price correlation by year:")
    print(corr_by_year.to_string())
    corr_by_year.to_csv(table_dir / "residual_load_price_corr_by_year.csv")

    # -----------------------------------------------------------------
    # 5. Descriptive pre/post market-design comparison
    #    (NOT a causal claim -- the periods differ in year, fuel prices,
    #    demand, renewables penetration, weather, and more, not just MTU
    #    granularity. Both periods are hourly in this dataset; "post"
    #    just means the underlying auction itself cleared at 15-min
    #    granularity before being aggregated here.)
    # -----------------------------------------------------------------
    by_regime = df.groupby("post_15min_mtu")["price_eur_mwh"].agg(["mean", "median", "std", "min", "max", "count"])
    print("\n" + "=" * 70)
    print("DESCRIPTIVE PRE/POST MARKET-DESIGN COMPARISON")
    print("(0 = pre-15-min-MTU market design, 1 = post-15-min-MTU, aggregated to hourly here)")
    print("Descriptive only -- does NOT estimate a causal effect of the Oct-2025 change;")
    print("the two periods differ in far more than market design (fuel prices, demand, etc.)")
    print("=" * 70)
    print(by_regime.to_string())
    by_regime.to_csv(table_dir / "price_by_market_design_regime.csv")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    logger.info("Saving figures to %s", fig_dir)

    # Figure 1: full price history, spikes and negative prices visible
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp_utc"], price, linewidth=0.4, color="steelblue")
    ax.axhline(0, color="red", linewidth=0.6, linestyle="--")
    ax.set_title("DE-LU Day-Ahead Price History (development sample)")
    ax.set_ylabel("EUR/MWh")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_price_history.png", dpi=150)
    plt.close(fig)

    # Figure 2: price distribution (histogram, log-scaled y for tail visibility)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(price.dropna(), bins=200, color="steelblue")
    ax.set_yscale("log")
    ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_title("DE-LU Price Distribution (log-scaled y for tail visibility)")
    ax.set_xlabel("EUR/MWh")
    ax.set_ylabel("Count (log scale)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_price_distribution.png", dpi=150)
    plt.close(fig)

    # Figure 3: seasonality -- mean price by hour of day, weekday vs weekend
    fig, ax = plt.subplots(figsize=(10, 5))
    for weekend_flag, label, color in [(0, "Weekday", "steelblue"), (1, "Weekend", "darkorange")]:
        subset = df[df["weekend"] == weekend_flag].groupby("hour_local")["price_eur_mwh"].mean()
        ax.plot(subset.index, subset.values, marker="o", label=label, color=color)
    ax.set_title("Mean DE-LU Price by Hour of Day: Weekday vs Weekend")
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Mean EUR/MWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_seasonality_hour_weekday.png", dpi=150)
    plt.close(fig)

    # Figure 4: residual load vs price -- quantile-binned median relationship,
    # not a raw scatter. European power prices have extreme spikes that turn
    # a raw scatter into a blue cloud with a few huge vertical outliers;
    # binning by residual-load quantile shows the empirical relationship
    # in a form that's actually interpretable ("as residual load rises
    # across the distribution, median price does X").
    plot_df = valid.copy()
    plot_df["residual_load_bin"] = pd.qcut(plot_df["residual_load_forecast_mw"], q=20, duplicates="drop")
    binned = (
        plot_df.groupby("residual_load_bin", observed=True)
        .agg(
            residual_load_median=("residual_load_forecast_mw", "median"),
            price_median=("price_eur_mwh", "median"),
            price_mean=("price_eur_mwh", "mean"),
            n=("price_eur_mwh", "size"),
        )
        .reset_index()
        .sort_values("residual_load_median")
    )
    binned.to_csv(table_dir / "residual_load_price_binned.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    sample = valid.sample(min(15000, len(valid)), random_state=42)
    ax.scatter(sample["residual_load_forecast_mw"], sample["price_eur_mwh"],
               s=2, alpha=0.15, color="lightsteelblue", label="Hourly observations (sampled)")
    ax.plot(binned["residual_load_median"], binned["price_median"],
            marker="o", color="darkorange", linewidth=2, label="Median price per residual-load ventile")
    ax.set_title(f"Residual Load Forecast vs Price\n(Pearson={pearson_corr:.2f}, Spearman={spearman_corr:.2f})")
    ax.set_xlabel("Residual load forecast (MW)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_residual_load_vs_price.png", dpi=150)
    plt.close(fig)

    # Figure 5: mean price by year (long-run trend, e.g. 2022 energy crisis visible)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_year.index.astype(str), by_year["mean"], color="steelblue")
    ax.set_title("Mean DE-LU Price by Year (development sample)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean EUR/MWh")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_mean_price_by_year.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Run manifest -- proves exactly what dataset generated these figures
    # -----------------------------------------------------------------
    manifest = {
        "input_file": str(in_path),
        "run_version": run_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eda_min_timestamp": df["timestamp_utc"].min().isoformat(),
        "eda_max_timestamp": df["timestamp_utc"].max().isoformat(),
        "eda_rows": len(df),
        "total_rows_in_input_file": n_total,
        "holdout_start": cfg["splits"]["holdout_start"],
        "holdout_excluded": not include_holdout,
        "holdout_rows_excluded": holdout_excluded_rows,
    }
    with open(table_dir / "eda_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_figures = len(list(fig_dir.glob("*.png")))
    n_tables = len(list(table_dir.glob("*.csv")))
    print("\n" + "=" * 70)
    print(f"Saved {n_figures} figures to {fig_dir}")
    print(f"Saved {n_tables} summary tables + manifest to {table_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
