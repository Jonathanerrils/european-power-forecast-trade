# Release notes — Energy Economics submission v2

Suggested tag:

`energy-economics-submission-v2`

Suggested release title:

**Energy Economics submission v2 — Forecast-to-Trade under Information Constraints**

Release target:

`submission/energy-economics-2026-09-21-v2`

Frozen source merge commit:

`a44a0256ff4e35e144fff9d966765120931e1ff0`

## What changed from v1

This version incorporates the second-round robustness review while preserving the frozen primary experiment.

- trailing-profile benchmark aligned to the full 212-day / 5,087-hour holdout using late-December 2025 price history;
- 3/7/10/14/21/28-day trailing-profile sensitivity added;
- S4 versus trailing-7 bootstrap contrast added;
- simple-profile value capture quantified;
- fixed-threshold frontier added for interpreting S3;
- January--March concentration of S2-S1 value quantified;
- Table 5 common-row versus selected-window coverage reconciled;
- 60-day uncertainty-window lower-grid-boundary limitation disclosed;
- Tier-1 XGBoost versus Tier-1 ElasticNet development comparison quantified;
- ENTSO-E createdDateTime/raw-cache limitation clarified;
- stale significance wording removed;
- euro typography and section-heading punctuation cleaned.

## Approved PDF fingerprint

Final reviewed 47-page Overleaf PDF SHA-256:

`8e63248e6576790b36bfb9ce3d3dd0e5cb2de1793a938742605fa95de4015246`

## Evidential note

The January--July 2026 holdout is exposed and must not be reused for tuning.
Post-hoc analyses remain supplementary and are explicitly separated from the
frozen pre-exposure sign-consistency result.
