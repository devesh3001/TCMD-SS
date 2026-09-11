# Measured model evolution

All listed metrics are historical development results on the same 1,184-case baseline benchmark (128 clean, 1,056 corruptions), not a new final holdout. Variants 1–3 are scoring ablations sharing one trained checkpoint; they are not three separately trained networks. The user-described 3.61M-parameter/DINOv2 baseline was not found and is not represented as reproduced evidence.

## VERSION 1

Symptom:
Pixel residual alone misses many synthetic anomalies.

Diagnosis:
Normal terrain variation competes with reconstruction residual.

Fix:
Use complementary masked reconstruction and localized residual aggregation.

Measured result:
AUROC 0.7323; TPR 0.3409; clean FPR 0.0625. Source: `outputs/ablation_results.csv`; historical 99th-percentile threshold.

## VERSION 2

Symptom:
Diffusion-only recall is low at the clean-calibrated threshold.

Diagnosis:
Pixel residual does not represent every semantic deviation.

Fix:
Add the existing independent DINO semantic memory branch as a baseline ablation.

Measured result:
AUROC 0.8207; TPR 0.5114; clean FPR 0.0547. Source: `outputs/ablation_results.csv`; historical 99th-percentile threshold.

## VERSION 3

Symptom:
Semantic addition leaves frequency and other deviations unresolved.

Diagnosis:
Different corruptions produce different branch responses.

Fix:
Evaluate the existing fixed diffusion/semantic/local-spectral/latent fusion.

Measured result:
AUROC 0.8893; TPR 0.6581; clean FPR 0.0547. Source: `outputs/ablation_results.csv`; historical 99th-percentile threshold.

## VERSION 4

Symptom:
The baseline fuses independent detectors and does not make generation central to semantic/spectral discrepancy.

Diagnosis:
A higher baseline AUROC alone does not establish the requested generative-discrepancy method or subtle six-family performance.

Fix:
Candidate two-seed, 15-step complementary reconstruction with pixel, texture, corresponding-token semantic and global directional spectral discrepancies; source-independent A/B calibration; 95th-percentile maximum-expert threshold.

Measured result:
Six component/protocol tests pass. A single normal training-image GPU probe took 3.3545 seconds, residual MAE 0.06196, mean seed disagreement 0.04516. These are smoke measurements, not anomaly metrics. Development evaluation is blocked by Windows Application Control and DINO runtime import failures. Architecture is NOT frozen; final holdout is NOT evaluated.

Negative evidence: removing the latent baseline branch produced AUROC 0.8916 versus 0.8893 for V3. Equal-weight baseline fusion measured 0.9011, but neither result validates the new discrepancy protocol. No optional terrain/multiscale/copy detector was claimed as tested.

## VERSION 5

Symptom:
The runtime blocks full generative-discrepancy experiments; an AUROC improvement alone may hide unacceptable clean false alarms.

Diagnosis:
The cached development experiment exposes a mismatch between the small clean calibration sample and new source observations. Removing diffusion also fails the intended central-generative architecture.

Fix:
Evaluate 13 discrete cached-score mean/max ablations. Fit continuous robust normalization on 71 clean calibration-A images, then set q95 thresholds on 57 source-disjoint calibration-B images. Keep the final holdout untouched.

Measured result:
DINO + spectral mean reaches development AUROC 0.9327, TPR 0.9081 and FPR 0.2500. This is rejected as a primary model: no diffusion and excessive false positives. Pixel + DINO + spectral mean reaches AUROC 0.8981, TPR 0.8030 and FPR 0.1797. No candidate is deployed. The historical equal-weight ablation remains 0.9011 AUROC / 0.0391 FPR under its different normal q99 protocol. These are inspected, non-final results. Source: results/development/cached_fusion_v5/.

## VERSION 6

Symptom:
Historical independent-expert results did not validate generative discrepancy on subtle faults.

Diagnosis:
The historical benchmark used different corruptions and stronger severities. A controlled six-family study was needed before claiming progress.

Fix:
Run the trained generator with four complementary masks, 15 DDIM steps and two seeds. Fit pixel/texture/directional-spectral discrepancy on 64 calibration-A normals and q95 thresholds on 64 source-disjoint calibration-B normals. Evaluate 16 consumed originals and 288 synthetic cases at .15/.30/.50.

Measured result:
Three-channel maximum fusion: AUROC 0.6291, TPR 0.1840, FPR 0.1875, F1 0.3081. Mean fusion: AUROC 0.6094, TPR 0.1111, FPR 0.0625. Inference averaged 0.399 seconds per image over 432 images, with peak allocated GPU memory 112.8 MiB. There is no DINO branch in this named partial ablation. Texture-map MAD collapsed to zero and produced poor combined localization. Source: results/development/generative_v6/.

## VERSION 7

Symptom:
Broad reconstruction mismatch diluted isolated faults; broad spectral discrepancies mixed terrain variation with periodic artifacts. Zero texture-map MAD distorted localization.

Diagnosis:
Image-global scores and near-zero map scales were not robust across normal terrain.

Fix:
Normalize residuals against local mismatch, score observed-versus-generated narrow spectral peaks, and use clean-data spread when map MAD is degenerate. Reuse the same reconstructions and clean calibration roles. Compare mean/max fusion on development only.

Measured result:
Refined mean fusion: AUROC 0.7181, TPR 0.4514, FPR 0.1250, F1 0.6190; paired anomaly exceeds clean in 74.31% of 288 pairs. Refined max fusion: AUROC 0.7099, TPR 0.4340, FPR 0.1250. Dead-pixel localization AUROC 0.8346 and IoU 0.4463. Copy-move localization remains near chance. This is a development-selected three-channel method, not a final result or validated DINO extension. Source: results/development/generative_v7/.
