# Submission readiness

The supplied marsps.pdf labels itself a practice problem statement. This repository contains development evidence; it is not a completed new final evaluation.

| Requested item | Current evidence | Status |
|---|---|---|
| Normal-only generative model | Existing diffusion checkpoint, training manifest/history | Completed bounded baseline |
| Flag and explain structural anomalies | Saved baseline scores, thresholds, heatmap examples | Synthetic development evidence |
| Executed notebook | notebooks/TCMD_SS_Final.ipynb and root notebook | 112 cells; all 77 code cells pass in review mode, including live diffusion reconstruction. Fresh training, full development inference and DINO remain off by default. |
| PDF and editable report | report/TCMD_SS_Report.pdf and .md | Development report |
| Three or more model versions | MODEL_EVOLUTION.md, seven explicitly labelled versions | Includes measured failures; first three share one checkpoint |
| New generative discrepancy | src/generative/, configs/generative.yaml | Three-channel V6/V7 development evaluated; DINO semantic extension pending |
| Untouched final result | 71 originals / three reserved sources | Not run; architecture not frozen |
| GitHub repository | [devesh3001/TCMD-SS](https://github.com/devesh3001/TCMD-SS) | Private repository |

## Reproduce the work completed in this revision

```powershell
.\.venv\Scripts\python scripts/evaluate_cached_fusion.py
.\.venv\Scripts\python -m unittest tests.test_revision tests.test_discrepancy tests.test_cached_fusion -v
.\.venv\Scripts\python -m compileall -q src scripts tests
.\.venv\Scripts\python scripts/build_verified_submission.py
```

The builder preserves existing deliverables by default. `--revise-draft` archives the current report/notebook before rebuilding. Numeric test collection currently fails because Windows Application Control blocks NumPy's `_pcg64` extension; DINO imports also fail. No security restriction was disabled.

## GitHub packaging

Selected audit summaries, split manifests, training history and evaluation results are included in Git. Raw images, model weights, checkpoints, caches and notebook archives stay local. The notebook includes saved outputs; live reconstruction requires the dataset and trained checkpoint. Code licensing status is recorded in `LICENSE.md`.

## Remaining model work

Restore an authorized working numerical runtime, run DINOv2 observed/reconstruction discrepancies, extend the completed six-family/localization study with DINO, improve calibration using more clean sources, run development ablations, freeze all selected artifacts, then evaluate the reserved final set once. The diagnostic 0.9327 AUROC is not an accepted generative model and its 25% FPR must always accompany that number.

## Latest measured generative study

`results/development/generative_v6/` and `generative_v7/` contain real GPU/cached experiments on the same 16 development originals. V7 reaches 0.7181 AUROC, 45.14% TPR and 12.5% FPR. This is not comparable to the older 0.9011 benchmark. Six measured explanation figures are in `results/figures/generative_v7/`. VAD4Space-inspired prevalence analysis uses no additional dataset and does not change thresholds.

See `PROJECT_DEFENCE.md` for explanations of the design, failure cases and limitations. Present actual contributions accurately; no claim of unaided authorship is made.
