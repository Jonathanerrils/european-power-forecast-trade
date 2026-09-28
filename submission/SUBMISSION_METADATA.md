# Energy Economics submission metadata

## Article

**Journal:** Energy Economics  
**Article type:** Research article

**Title**  
Forecast-to-Trade under Information Constraints: Frozen Out-of-Sample Evidence from the Germany–Luxembourg Day-Ahead Electricity Market

## Author

**Name:** Jonathan Anthonio Nii Pedro Nelson  
**Affiliation:** Kwame Nkrumah University of Science and Technology (KNUST), Kumasi, Ghana  
**Corresponding author:** Yes  
**Email:** jonathannelson707@gmail.com  
**ORCID:** 0009-0006-9393-6237

## Abstract

Electricity-price forecasts are commonly judged by statistical errors even though their operational value depends on the decisions they support. This study evaluates whether a forecasting advantage survives point-in-time information constraints and translates into economic value under a frozen out-of-sample protocol for the Germany–Luxembourg day-ahead market. Public ENTSO-E data from 2019–2025 are used for development, while January–July 2026 is reserved for a first-look holdout. The information set is split into a Full specification and a Tier-1 robustness specification that excludes wind- and solar-derived variables whose exact historical availability at the simulated 11:45 D-1 decision time cannot be verified. XGBoost is selected during development against naive and ElasticNet benchmarks, and uncertainty is estimated from delivery-day-safe empirical residual quantiles. Forecasts are then mapped into a pre-specified single-cycle storage decision rule with a frozen efficiency–cost sensitivity grid. On 5,087 holdout hours, Full XGBoost attains an MAE of €17.51/MWh versus €29.33/MWh for lag-24. Under the primary economic assumptions, the Full point strategy earns €20,002.02 compared with €18,564.42 for lag-24, and the incremental value remains positive in all nine pre-specified sensitivity cells. A post-hoc trailing seven-day hourly profile narrows the economic comparison substantially: on the same 212 holdout days, Full exceeds it by €279.73, with a 7-day block-bootstrap interval spanning zero. Across six exploratory trailing-profile windows (3, 7, 10, 14, 21 and 28 days), Full's margin ranges from approximately €279.73 to €492.44. The uncertainty-aware Full strategy implements a day-varying abstention threshold; it sacrifices €872.01 of aggregate P&L but raises the trade hit rate from 91.9% to 99.4% and improves the worst day from -€17.55 to -€0.83. The contribution is therefore methodological and evidential rather than algorithmic: forecast value is assessed under explicit information-timing boundaries, delivery-day-safe uncertainty, frozen pre-exposure rules, economic robustness and tail-risk diagnostics.

## Keywords

1. electricity price forecasting
2. Germany–Luxembourg
3. battery arbitrage
4. forecast uncertainty
5. economic value
6. out-of-sample evaluation

## Highlights

- Frozen holdout links electricity-price accuracy to realized storage value
- Full XGBoost cuts holdout MAE by 40% versus the lag-24 benchmark
- Forecast-value gains remain positive across all nine frozen cost-efficiency cells
- Uncertainty acts as a day-varying abstention threshold, reducing downside risk
- Post-hoc simple profiles capture most of the economic gain despite worse MAE

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Competing interests

The author declares no competing interests.

## Data and code availability

The code, frozen research protocols, model artifacts, machine-readable results and supplementary robustness analyses supporting this study are archived on Zenodo at https://doi.org/10.5281/zenodo.22878011. The exact derived feature-matrix snapshot used for the reported analyses is archived separately at https://doi.org/10.5281/zenodo.22883513. The dataset metadata are public while the archived files are restricted because underlying source data originate from the ENTSO-E Transparency Platform and remain subject to applicable source-data rights and reuse conditions. The public source repository is https://github.com/Jonathanerrils/european-power-forecast-trade.

## Generative AI declaration

During the preparation of this work, the author used ChatGPT (OpenAI) to assist with language editing, manuscript organization, LaTeX preparation and review of computational documentation. After using this tool, the author reviewed and edited the content as needed, independently verified the reported numerical results against the preserved research artifacts, and takes full responsibility for the content of the publication.

## Graphical abstract

Not planned for initial submission. Current Elsevier guidance treats a graphical abstract as optional. If the journal submission portal specifically requests one, prepare and upload it as a separate file.

## Submission files

- Manuscript PDF: GitHub Actions artifact from the final submission branch
- Manuscript source: `manuscript/main.tex` plus referenced section, figure and bibliography files
- Highlights: `submission/HIGHLIGHTS.txt`
- Cover letter: `submission/COVER_LETTER.txt`
- Replication guide: `REPLICATION.md`
- Final freeze record: `submission/FINAL_FREEZE_RECORD.md`
