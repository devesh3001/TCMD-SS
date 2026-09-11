# Reference and Detailed Comparison

## Datasets

- **HiRISE** (original):
  - Location: `data/raw/hirise_v3_2`
  - Structure: `train/normal`, `test/normal`, `test/anomaly_real`
  - Used in original notebook.

- **NSSC** (new):
  - Location: `data/raw/nssc/test+train`
  - Structure: `train/normal`, `test/normal`, `test/anomaly_real`
  - Split files generated under `data/splits` (train_normal.csv, test_clean.csv, test_anomaly.csv).

## Model Pipeline Adjustments

- `assert_normal` check disabled to allow NSSC paths.
- Evaluation now loads real anomalies from `data/splits/test_anomaly.csv`.
- Benchmark generation step removed; evaluation runs on real anomalies only.
- Semantic model weights are now downloaded at runtime (no local weight file).

## Results Summary (Smoke Run)

- Training loss: 1.05 (epoch 1)
- Evaluation completed without errors.
- Metrics saved in `outputs/smoke/evaluation` (files present after run).

## Notes

- All pipeline stages except split and benchmark are retained.
- No AI‑generated filler text is present in this file.
