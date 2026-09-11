# Revision status — incomplete

- Existing baseline and audit preserved
- Exposure-aware source/group-disjoint manifests reserved
- Candidate generative discrepancy and normal-only robust calibration implemented
- Exclusive final-run freeze utility implemented
- Six new tests passed
- Two-seed GPU reconstruction smoke passed

## Measured baseline

AUROC 0.8893, AUPRC 0.9852, TPR 0.6581, FPR 0.0547, F1 0.7907, precision 0.9900. These are development results from the saved historical protocol, not the requested new final benchmark. The trained network has 570,497 parameters, completed eight epochs/1,024 optimizer steps, and selected best normal-validation MAE 0.10808.

## Reserved protocol

| Role | Images | Sources |
|---|---:|---:|
| train_normal | 39997 | 158 |
| calibration_a | 1418 | 22 |
| calibration_b | 959 | 19 |
| development | 1538 | 30 |
| final_holdout | 71 | 3 |

All source/base/duplicate-linked intersections are empty. Final eligibility is conditional on workspace exposure history: externally inspected images cannot be excluded without their IDs. Three final observations are too few for strong generalization claims.

## Runtime blocker

- NumPy numpy.random._pcg64: Windows Application Control blocks native extension; official NumPy reinstall did not resolve it.
- timm import: torch._dynamo.utils NP_SUPPORTED_MODULES import failure; DINO cannot currently load.

A trusted working Python environment or administrator resolution of the native-extension policy is needed before continuing. No security policy was disabled. Full pytest failed during collection; six isolated new tests and compileall passed.

## Remaining work

- Six-family development benchmark integration
- DINOv2 discrepancy evaluation
- Independent calibration and ablation runs for revision
- Architecture selection and freeze
- One-time final evaluation
- Final competition notebook and report

Resume from `configs/generative.yaml` and the reserved manifests. Do not rerun the historical baseline onto its existing outputs. Do not claim or evaluate final until development is complete and all inputs, calibration artifacts and source/config files are frozen.
