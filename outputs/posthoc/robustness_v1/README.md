# Post-hoc robustness diagnostics v1

These diagnostics were introduced **after exposure to the January--July 2026 holdout**. They are supplementary analyses and do not redefine, replace, or retroactively extend the frozen primary sign-consistency criterion.

The analyses include:
- a weekday-aware naive benchmark (lag-24 Tuesday--Friday; lag-168 Saturday--Monday);
- a trailing hourly mean-price profile, with the 7-day window as the primary descriptive post-hoc benchmark and 3/10/14/21/28-day windows reported as an exploratory sensitivity family;
- 7-day circular moving-block bootstrap intervals with 20,000 replicates and fixed seed 20260920;
- monthly decomposition of the frozen S2--S1 contrast;
- holdout interval coverage, miss rates, widths and Winkler scores overall, by month and by local delivery hour;
- a fixed-threshold frontier to interpret how much of S3's behavior is attributable to a roughly constant participation hurdle.

The trailing-profile benchmark is evaluated on all 212 holdout days. For 1--7 January 2026 it is initialized from realized price profiles in late December 2025, which were already public before the corresponding 2026 D-1 decision origins. This removes an unnecessary artificial burn-in while preserving temporal admissibility.

Primary frozen artifacts remain under `outputs/holdout/delu_features/`. The post-hoc analyses in this folder were not part of the original frozen protocol.
