# Energy Economics submission checklist

## Scientific manuscript
- [x] Frozen Jan--Jul 2026 primary experiment preserved.
- [x] Full/Tier-1 point-in-time boundary stated explicitly.
- [x] Uncertainty layer derived correctly as a day-varying abstention threshold.
- [x] Nine-cell result described as a sign-consistency sensitivity criterion, not independent confirmation.
- [x] Stronger post-hoc simple benchmarks added and clearly labelled.
- [x] 7-day moving-block bootstrap diagnostics added and clearly labelled post hoc.
- [x] Holdout uncertainty calibration diagnostics added.
- [x] Economic P&L normalization stated as a 1-MWh daily opportunity.
- [x] Grid-edge hyperparameter selections disclosed.
- [x] All nine figures referenced in the prose.
- [x] All manuscript tables referenced in the prose.
- [x] Citation keys and cross-references internally consistent.
- [x] No “preregistered” wording remains.

## Replication package
- [x] Frozen protocol commits identified.
- [x] Holdout hourly inputs preserved in machine-readable form.
- [x] Daily holdout strategy results preserved.
- [x] Holdout point-forecast table preserved.
- [x] Frozen sensitivity results preserved.
- [x] Tail-risk outputs preserved.
- [x] Post-hoc robustness script and core outputs preserved.
- [x] Manuscript-to-artifact map added in REPLICATION.md.
- [ ] Archive the exact final derived feature dataset used for the manuscript, subject to ENTSO-E terms.
- [ ] Record SHA-256 checksum of the archived feature dataset.
- [ ] Create a stable repository release for the submission version.
- [ ] Archive the submission release on Zenodo or equivalent and record its DOI.
- [ ] Verify a clean-environment rerun of the replication instructions.

## Journal-facing files
- [x] Manuscript in Elsevier elsarticle format.
- [x] Short cover letter drafted.
- [x] Five Highlights drafted; each is <=85 characters.
- [x] Keywords present in manuscript.
- [x] Competing-interest declaration present.
- [x] Funding declaration present.
- [x] Data/code availability section present.
- [x] Generative-AI declaration updated to current Elsevier policy.
- [x] Final acknowledgements text included.
- [ ] Verify whether a graphical abstract is requested/desired at actual submission.
- [ ] Confirm current submission-system metadata immediately before upload.
- [x] Final author name, KNUST affiliation and corresponding-author details confirmed.

## Final document QA
- [x] Latest manuscript changes reflected in Overleaf.
- [x] Final manuscript compiled successfully in Overleaf.
- [x] Bibliography resolves in the final compile.
- [x] Final PDF visually checked for missing figures, table fit and appendix rendering.
- [x] All nine figures checked for legibility.
- [x] Page breaks and float placement reviewed in the final PDF.
- [x] Final PDF exported and key reported results checked against preserved artifacts.
- [ ] Freeze final submission commit/tag only after all checks pass.

## Submission discipline
The January--July 2026 holdout is already exposed. Do not use it to tune new models, thresholds, uncertainty methods or information sets. Any redesigned method should be frozen before evaluation on a new untouched period.
