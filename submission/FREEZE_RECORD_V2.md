# Energy Economics submission freeze — revised v2

Date: 2026-09-21

## Manuscript

Title: Forecast-to-Trade under Information Constraints: Frozen Out-of-Sample Evidence from the Germany--Luxembourg Day-Ahead Electricity Market

Author: Jonathan Anthonio Nii Pedro Nelson

Affiliation: Kwame Nkrumah University of Science and Technology (KNUST), Kumasi, Ghana

Target journal: Energy Economics

## Repository freeze

Second-round robustness revision merged via pull request #4.

Main merge commit:

`a44a0256ff4e35e144fff9d966765120931e1ff0`

Submission freeze branch:

`submission/energy-economics-2026-09-21-v2`

Earlier pre-review submission freeze:

`submission/energy-economics-2026-09-20`

The present v2 freeze supersedes that earlier submission snapshot for journal submission.

Pre-holdout protocol commit:

`a2d1f9a79022ca1e2731d52803061825038c69f5`

Final Full and Tier-1 model-fit commit:

`54a7c2e77122134eca31a09798cc1614df1c634c`

## Approved compiled PDF

Approved Overleaf compile: 47 pages

SHA-256:

`8e63248e6576790b36bfb9ce3d3dd0e5cb2de1793a938742605fa95de4015246`

The approved PDF includes:
- full 212-day post-hoc trailing-profile benchmark alignment;
- 3/7/10/14/21/28-day profile-window sensitivity;
- S4 versus trailing-7 block-bootstrap inference;
- fixed-threshold frontier diagnostics;
- Table 5 row-set reconciliation;
- 60-day uncertainty-window boundary disclosure;
- Tier-1 XGBoost versus Tier-1 ElasticNet development comparison;
- corrected euro typography and final manuscript acknowledgements.

## Evidential boundary

The January--July 2026 holdout has already been exposed. No further tuning of
models, information sets, uncertainty rules, thresholds, benchmark windows or
economic decision rules should use this holdout. Analyses under
`outputs/posthoc/robustness_v1/` are supplementary and do not redefine the
frozen primary sign-consistency criterion.

## Remaining archival tasks

- Create a formal GitHub release/tag from this v2 submission freeze.
- Archive the release on Zenodo or an equivalent repository and record the DOI.
- Archive the exact final derived feature dataset, subject to ENTSO-E terms,
  and record its SHA-256 checksum.
- Verify the replication instructions in a clean environment before journal
  submission.
