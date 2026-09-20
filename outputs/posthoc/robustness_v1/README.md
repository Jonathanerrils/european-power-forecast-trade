# Post-hoc robustness diagnostics v1

These diagnostics were introduced **after exposure to the January--July 2026 holdout**. They are supplementary analyses and do not redefine, replace, or retroactively extend the frozen primary confirmation criterion.

The analyses include:
- a weekday-aware naive benchmark (lag-24 Tuesday--Friday; lag-168 Saturday--Monday);
- a trailing seven-completed-day hourly mean price profile, evaluated from holdout day 8 onward;
- 7-day circular moving-block bootstrap intervals with 20,000 replicates and seed 20260920;
- monthly decomposition of the frozen S2-S1 contrast;
- holdout interval coverage, miss rates, widths and Winkler scores overall, by month and by local delivery hour.

Primary frozen artifacts remain under `outputs/holdout/delu_features/`.
