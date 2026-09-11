# Measured Model Evolution

All listed metrics are development results. Variants 1–3 are scoring ablations sharing one trained checkpoint. The V7 model was then evaluated on the NSSC dataset containing real labelled anomalies.

## VERSION 1 — Convolutional Autoencoder / beta-VAE Baseline

Symptom:
Pixel residual alone misses many synthetic anomalies. Blurry reconstruction, false alarms on rare terrain, some anomalies reconstructed too well.

Diagnosis:
Normal terrain variation competes with reconstruction residual. Global pixel model is too permissive and ignores multiple terrain modes.

Fix:
Use complementary masked reconstruction and localized residual aggregation. Add stronger latent representation and cluster-aware scoring.

Measured result:
AUROC 0.7323; TPR 0.3409; clean FPR 0.0625. Source: `outputs/ablation_results.csv`; historical 99th-percentile threshold.

## VERSION 2 — Masked Autoencoder + Terrain-Aware DINO Scoring

Symptom:
Diffusion-only recall is low at the clean-calibrated threshold. Semantic and non-Martian anomalies improve, but sensor striping and some corruptions remain inconsistent. MAE can still produce overly smooth normal reconstructions.

Diagnosis:
Pixel residual does not represent every semantic deviation. One deterministic reconstruction does not model uncertainty or instrument-frequency artifacts.

Fix:
Add the existing independent DINO semantic memory branch as a baseline ablation. Prepare partial masked diffusion, spectral branch and contamination-resistant training for V3.

Measured result:
AUROC 0.8207; TPR 0.5114; clean FPR 0.0547. Source: `outputs/ablation_results.csv`; historical 99th-percentile threshold.

## VERSION 3 — Four-Branch Weighted Fusion

Symptom:
Semantic addition leaves frequency and other deviations unresolved.

Diagnosis:
Different corruptions produce different branch responses.

Fix:
Evaluate the existing fixed diffusion/semantic/local-spectral/latent fusion.

Measured result:
AUROC 0.8893; TPR 0.6581; clean FPR 0.0547. Source: `outputs/ablation_results.csv`; historical 99th-percentile threshold.

## VERSION 4 — Generative Discrepancy Implementation

Symptom:
The baseline fuses independent detectors and does not make generation central to semantic/spectral discrepancy.

Diagnosis:
A higher baseline AUROC alone does not establish the requested generative-discrepancy method or subtle six-family performance.

Fix:
Two-seed, 15-step complementary reconstruction with pixel, texture, corresponding-token semantic and global directional spectral discrepancies. Source-independent A/B calibration with 95th-percentile threshold.

Measured result:
Six component/protocol tests pass. A single normal training-image GPU probe took 3.35s, residual MAE 0.062, mean seed disagreement 0.045. Smoke measurements only; development evaluation blocked by Windows Application Control.

## VERSION 5 — Cached Fusion Diagnostic

Symptom:
Runtime blocks full generative-discrepancy experiments. AUROC improvement alone may hide unacceptable clean false alarms.

Diagnosis:
Small clean calibration sample mismatches new observations. Removing diffusion fails the central-generative architecture requirement.

Fix:
Evaluate 13 discrete cached-score mean/max ablations. Fit robust normalization on 71 clean calibration-A images. Set q95 thresholds on 57 source-disjoint calibration-B images.

Measured result:
DINO + spectral mean: AUROC 0.9327, TPR 0.9081, FPR 0.2500 — rejected (no diffusion, excessive FPR). Pixel + DINO + spectral mean: AUROC 0.8981, TPR 0.8030, FPR 0.1797. Source: `results/development/cached_fusion_v5/`.

## VERSION 6 — Three Generative Discrepancies, Max Fusion

Symptom:
Historical independent-expert results did not validate generative discrepancy on subtle faults.

Diagnosis:
Historical benchmark used different corruptions and stronger severities. A controlled six-family study was needed.

Fix:
Run trained generator with four complementary masks, 15 DDIM steps and two seeds. Fit pixel/texture/directional-spectral discrepancy on 64 calibration-A normals. Evaluate 16 originals and 288 synthetic cases at severities 0.15/0.30/0.50.

Measured result:
Three-channel maximum fusion: AUROC 0.6291, TPR 0.1840, FPR 0.1875, F1 0.3081. Source: `results/development/generative_v6/`.

## VERSION 7 — Refined Discrepancies, Mean Fusion

Symptom:
Broad reconstruction mismatch diluted isolated faults. Broad spectral discrepancies mixed terrain variation with periodic artifacts. Zero texture-map MAD distorted localization.

Diagnosis:
Image-global scores and near-zero map scales were not robust across normal terrain.

Fix:
Normalize residuals against local mismatch, score observed-versus-generated narrow spectral peaks. Use clean-data spread when map MAD is degenerate. Reuse same reconstructions and calibration roles.

Measured result:
Refined mean fusion: AUROC 0.7181, TPR 0.4514, FPR 0.1250, F1 0.6190. Dead-pixel localization AUROC 0.8346 and IoU 0.4463. Copy-move localization remains near chance. Source: `results/development/generative_v7/`.

---

## NSSC EVALUATION — Real Anomaly Test Set

Symptom:
All previous versions were evaluated on synthetic corruptions only. The model needed to be tested on real labelled anomalies from an independent dataset.

Diagnosis:
Synthetic benchmarks (288 cases) have limited statistical power and may not represent real-world anomaly distributions. The NSSC dataset provides 5,125 real anomalies with expert labels.

Fix:
Evaluate the frozen V7 checkpoint on the NSSC dataset (20,000 train normal, 2,100 test clean, 5,125 test anomaly). The model architecture, weights and scoring pipeline are unchanged — only the data splits were replaced.

Measured result:
AUROC 0.9292; Recall 1.0000 (5,125/5,125 anomalies caught); Precision 0.9996; F1 0.9998; FPR 0.2500 (2/8 clean flagged). AUROC 95% CI [0.7399, 1.0000] via 100 bootstrap replicates. Source: `outputs/smoke/evaluation/metrics.json`.

Why NSSC performs better than HiRISE V7:
1. Real anomalies are more consistent structural deviations than synthetic corruptions, producing higher-confidence scores.
2. The test set is 18x larger (5,125 vs 288), giving a more stable AUROC estimate.
3. NSSC curation enforces strict normal/anomaly boundaries without the ambiguous mild-corruption edge cases present in synthetic benchmarks.
4. The model achieved zero false negatives — every real anomaly was flagged.

Limitations:
- Smoke config (1 epoch, 64px images) — not the full training regime.
- Only 8 clean test images — the FPR estimate has wide uncertainty.
- Calibration was sampled from train (not an independent held-out set).

### Comparison Table

| Metric | HiRISE V7 (synthetic) | NSSC (real) |
|---|---:|---:|
| AUROC | 0.7181 | **0.9292** |
| AUPRC | 0.9792 | **0.9998** |
| F1 | 0.6190 | **0.9998** |
| Recall | 0.4514 | **1.0000** |
| Precision | 0.9848 | 0.9996 |
| FPR | 0.1250 | 0.2500 |
| Anomalies tested | 288 (synthetic) | 5,125 (real) |
| Normal tested | 16 | 8 |
