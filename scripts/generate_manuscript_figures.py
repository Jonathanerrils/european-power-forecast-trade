"""Generate publication PNG figures 5--9 for the manuscript.

Figures 1--4 are the corrected EDA figures already preserved under outputs/eda.
This script makes PNG assets the canonical manuscript representation for
Figures 5--9; no TikZ/PGFPlots figure source is required.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 220


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# Figure 5: forecast-to-trade workflow
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

def box(x, y, w, h, text, fontsize=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fill=False)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="->", mutation_scale=12))

box(0.3, 5.2, 1.8, 0.9, "Public ENTSO-E data\nPrices, load,\nwind and solar")
box(2.5, 5.2, 2.0, 0.9, "Reconstruction & cleaning\nA03, product selection,\nDST, hourly alignment")
box(5.0, 5.8, 1.8, 0.75, "Full information set\n21 predictors")
box(5.0, 4.65, 1.8, 0.75, "Tier-1 information set\n18 predictors")
box(7.3, 5.2, 1.8, 0.9, "Forecasting models\nNaive, ElasticNet,\nXGBoost")
box(9.6, 5.2, 1.9, 0.9, "Chronological\ndevelopment\n2019–2025")
box(9.6, 3.4, 1.9, 0.9, "Day-origin-safe\nuncertainty\n60-day residual window")
box(7.3, 3.4, 1.8, 0.9, "Frozen economic rules\nS0–S5 and\n3×3 sensitivity grid")
box(5.0, 3.4, 1.8, 0.9, "First-look holdout\nJan–Jul 2026")
box(3.7, 1.4, 1.8, 0.75, "Forecast accuracy")
box(5.6, 1.4, 1.8, 0.75, "Economic value")
box(7.5, 1.4, 1.8, 0.75, "Tail risk")
arrow(2.1, 5.65, 2.5, 5.65)
arrow(4.5, 5.65, 5.0, 6.15)
arrow(4.5, 5.65, 5.0, 5.02)
arrow(6.8, 6.15, 7.3, 5.85)
arrow(6.8, 5.02, 7.3, 5.45)
arrow(9.1, 5.65, 9.6, 5.65)
arrow(10.55, 5.2, 10.55, 4.3)
arrow(9.6, 3.85, 9.1, 3.85)
arrow(7.3, 3.85, 6.8, 3.85)
arrow(5.9, 3.4, 4.6, 2.15)
arrow(5.9, 3.4, 6.5, 2.15)
arrow(5.9, 3.4, 8.4, 2.15)
ax.add_patch(Rectangle((7.05, 3.15), 4.7, 1.4, fill=False,
                       linestyle="--", linewidth=1.3))
ax.text(9.4, 4.62, "Protocol frozen before holdout exposure",
        ha="center", va="bottom", fontsize=9)
ax.set_title("Forecast-to-trade research workflow", fontsize=12, pad=10)
save(fig, "fig5_forecast_to_trade_workflow.png")


# Figure 6: chronological design
fig, ax = plt.subplots(figsize=(11.8, 5.7))
ax.set_xlim(2018.7, 2026.82)
ax.set_ylim(0.35, 5.85)
rows = [
    ("Fold 1", 2019.00, 2023.00, 2023.00, 2024.00),
    ("Fold 2", 2019.00, 2024.00, 2024.00, 2025.00),
    ("Fold 3", 2019.00, 2025.00, 2025.00, 2025.75),
    ("Regime stress", 2019.00, 2025.75, 2025.75, 2026.00),
]
ys = [4.65, 3.65, 2.65, 1.65]
h = 0.44
for (label, tr0, tr1, va0, va1), y in zip(rows, ys):
    ax.add_patch(Rectangle((tr0, y), tr1-tr0, h,
                           fill=False, hatch="///", linewidth=1.0))
    ax.add_patch(Rectangle((va0, y), va1-va0, h,
                           fill=False, hatch="xx", linewidth=1.0))
    ax.text(2018.94, y+h/2, label, ha="right", va="center", fontsize=9)

ax.axvline(2026.0, linestyle="--", linewidth=1.4)
ax.text(2026.0, 5.36, "Protocol freeze\nand final refit",
        ha="center", va="bottom", fontsize=8.5)

holdout_start, holdout_end, hold_y = 2026.00, 2026.58, 0.83
ax.add_patch(Rectangle((holdout_start, hold_y), holdout_end-holdout_start, h,
                       fill=False, hatch="..", linewidth=1.2))
ax.text((holdout_start+holdout_end)/2, hold_y+h/2, "Holdout",
        ha="center", va="center", fontsize=8.5)
ax.annotate("1 Jan–31 Jul 2026",
            xy=((holdout_start+holdout_end)/2, hold_y+h),
            xytext=(2026.34, 1.62),
            ha="center", va="bottom", fontsize=8.5,
            arrowprops=dict(arrowstyle="-", linewidth=0.8))

ax.text(2022.45, 0.47,
        "All model, uncertainty and economic-rule choices use 2019–2025 only.",
        ha="center", va="center", fontsize=8.8)
legend_handles = [
    Patch(fill=False, hatch="///", label="Training"),
    Patch(fill=False, hatch="xx", label="Validation / stress"),
    Patch(fill=False, hatch="..", label="Frozen holdout"),
]
ax.legend(handles=legend_handles, loc="upper left",
          bbox_to_anchor=(0.02, 0.99), ncol=3, frameon=False, fontsize=8.5)
ax.set_xticks(range(2019, 2027))
ax.set_yticks([])
ax.set_xlabel("Delivery year")
ax.set_title("Chronological development and frozen holdout design",
             fontsize=11.5, pad=10)
for spine in ["left", "right", "top"]:
    ax.spines[spine].set_visible(False)
save(fig, "fig6_chronological_evaluation_design.png")


# Figure 7: holdout forecast accuracy
models = ["Lag-24", "Lag-168", "XGB Full", "XGB Tier-1"]
mae = [29.3291, 36.0141, 17.5122, 24.3606]
rmse = [46.8553, 56.5983, 29.8691, 38.4137]
x = np.arange(len(models))
width = 0.36
fig, ax = plt.subplots(figsize=(9, 6))
b1 = ax.bar(x - width/2, mae, width, label="MAE")
b2 = ax.bar(x + width/2, rmse, width, label="RMSE")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Error (EUR/MWh)")
ax.set_title("Frozen 2026 holdout forecast accuracy")
ax.legend()
ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=8)
ax.text(0.5, -0.13, "N = 5,087 common hourly observations",
        transform=ax.transAxes, ha="center", fontsize=9)
save(fig, "fig7_holdout_forecast_accuracy.png")


# Figure 8: cumulative holdout P&L
pnl_path = ROOT / "manuscript" / "data" / "holdout_cumulative_pnl.csv"
pnl = pd.read_csv(pnl_path)
pnl["delivery_date"] = pd.to_datetime(pnl["delivery_date"])
fig, ax = plt.subplots(figsize=(10, 6.5))
for col, label in [
    ("S1", "S1 Lag-24"),
    ("S2", "S2 Full"),
    ("S3", "S3 Full + uncertainty"),
    ("S4", "S4 Tier-1"),
    ("S5", "S5 Tier-1 + uncertainty"),
]:
    ax.plot(pnl["delivery_date"], pnl[col], label=label, linewidth=1.8)
ax.set_xlabel("Delivery date")
ax.set_ylabel("Cumulative net P&L (EUR)")
ax.set_title("Cumulative net P&L in the frozen 2026 holdout")
ax.legend(fontsize=8)
ax.grid(True, linestyle=":", linewidth=0.6)
fig.autofmt_xdate()
save(fig, "fig8_holdout_cumulative_pnl.png")


# Figure 9: sensitivity heatmap
grid = np.array([
    [1204.42325, 1050.04900, 1050.95750],
    [1386.136875, 1437.596375, 1364.73050],
    [1439.21350, 1432.42320, 1449.80800],
])
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(grid, aspect="auto")
ax.set_xticks([0, 1, 2], labels=["5", "10", "15"])
ax.set_yticks([0, 1, 2], labels=["0.70", "0.85", "0.92"])
ax.set_xlabel("Degradation parameter c (EUR)")
ax.set_ylabel("Round-trip efficiency")
ax.set_title("Holdout sensitivity of S2 − S1 net P&L")
for i in range(3):
    for j in range(3):
        label = f"{grid[i,j]:,.0f}"
        if i == 1 and j == 1:
            label += "\nPrimary"
        ax.text(j, i, label, ha="center", va="center", fontsize=9)
ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False, linewidth=2))
fig.colorbar(im, ax=ax, label="Incremental net P&L (EUR)")
ax.text(0.5, -0.16, "Positive in all 9/9 pre-specified cells",
        transform=ax.transAxes, ha="center", fontsize=9)
save(fig, "fig9_holdout_sensitivity_heatmap.png")

print("Generated manuscript PNG figures 5--9 in", OUT)
