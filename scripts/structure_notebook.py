"""Keep the research narrative separate from optional experiments."""

def structure_notebook(nb):
    if nb.metadata.get('tcmdss', {}).get('structured'):
        return nb
    cells = nb.cells
    # Guard the source layout so a later builder change cannot silently move wrong cells.
    assert len(cells) == 120 and 'Optional DINO' in cells[63].source
    text = {
        0: '# TCMD-SS: HiRISE anomaly detection\n\nMasked conditional diffusion for detecting structural image faults. All eight terrain classes are normal.\n\n**V7 development AUROC: 0.7181. Final evaluation: pending.**',
        1: '## Contents\n\n1. Setup\n2. Dataset and splits\n3. Model\n4. Training and reconstruction\n5. Development anomalies\n6. Anomaly scores\n7. Calibration and inference\n8. Results\n9. Localization and failure cases\n10. Checks and conclusions\n\nAppendix: historical results, prevalence analysis and optional DINO code.',
        2: '## 1. Setup\n\nAttach the project folder, saved checkpoint and extracted HiRISE dataset. Enable a GPU in Kaggle or Colab and set the path overrides below if needed. The default mode uses saved results and runs a reconstruction demo. Training and full inference are optional.',
        9: '### Display helpers',
        11: '## 2. Dataset and splits\n\nThe dataset is NASA/JPL [HiRISE v3.2](https://zenodo.org/records/4002935). The following tables use the completed audit and download verification.',
        15: 'Class imbalance and repeated augmentations can bias sampling. Bounded experiments use originals, with source and duplicate-linked groups kept together.',
        16: '### Augmentation and duplicate checks\n\nSuffixes identify rotations (`r90`, `r180`, `r270`), flips (`fh`, `fv`) and brightness changes (`brt`). The parser illustrates this naming pattern; duplicate counts come from the full audit.',
        19: '### Split checks\n\nTraining, calibration A, calibration B, development and final holdout are separated by source and linked image groups. Calibration A fits normalization; calibration B sets the threshold. Final images are not loaded.',
        22: '### Image loading and examples\n\nPaths are resolved against the selected dataset folder. The loader checks the normal-only training labels.',
        27: '## 3. Model\n\nThe U-Net receives noisy hidden pixels, visible context and a binary mask, and predicts diffusion noise. Group normalization supports small batches.',
        38: '### Reconstruction\n\nFour complementary masks reconstruct every pixel while hidden. Two fixed noise seeds produce an average reconstruction and a disagreement map.',
        41: '## 4. Training and reconstruction\n\nThe training loop uses the model and loss above and saves to a new folder. The historical checkpoint was trained with `scripts/train_diffusion.py`.',
        44: '### Training history and checkpoint\n\nThe saved run completed eight epochs. Its best checkpoint was selected on a small normal validation subset.',
        48: '## 5. Development anomalies\n\nSix synthetic fault families use severities 0.15, 0.30 and 0.50. These images are excluded from generator training. Masks identify the modified regions; foreign content is generated procedurally.',
        53: '## 6. Anomaly scores\n\nPixel residual, texture deficiency and directional spectral differences compare the observed image with its reconstruction.',
        58: '### V7 refinement\n\nLocal residual normalization reduces broad reconstruction mismatch. Narrow-band scoring emphasizes periodic artifacts, and a clean-data spread estimate handles zero MAD.',
        73: '### Development inference\n\nEnable `RUN_DEVELOPMENT_INFERENCE` to score new development cases. The loop uses clean calibration A/B and writes predictions to a separate run folder.',
        76: '## 8. Results\n\nAUROC measures ranking; recall and false-positive rate use the calibrated threshold. Average precision depends on anomaly prevalence. The following cells recompute metrics from saved predictions.',
        82: '### V6 and V7 comparison\n\nBoth versions use the same development cases. The historical benchmark in the appendix uses different corruptions.',
        85: '### Results by fault and severity\n\nCorruptions of the same original are dependent; 288 synthetic cases do not represent 288 independent sources.',
        88: '## 9. Localization and failure cases\n\nPixel calibration is separate from image-score calibration. Spectral maps show row/column attribution, not inverse-FFT localization. Full-image stripe masks have no negative pixels, so their pixel AUROC is undefined.',
        91: '### Example predictions\n\nEach example shows the input, reconstruction, uncertainty and discrepancy maps. Heatmaps have separate display scales. All examples below are saved severity-0.30 cases.',
        105: '## Appendix A. Historical benchmark\n\nThis earlier study used 128 clean images and 1,056 different synthetic cases. Its equal-weight AUROC of 0.9011 is not directly comparable to V7.',
        107: '## Appendix B. Rare-anomaly sensitivity\n\nReweight the historical errors to a 5% anomaly prevalence. These are analytical estimates, not new deployment measurements. Related work: [VAD4Space](https://arxiv.org/abs/2603.13993).',
        63: '## Appendix C. Optional DINO extension\n\nFrozen DINOv2 compares corresponding patches in the observed and generated images. This extension is not part of the V7 results. `RUN_DINO=True` requires a working timm runtime and may download pretrained weights.',
        110: '## 10. Checks and conclusions\n\nCheck group separation, mask coverage, tied-score metrics, calibration spread and corruption reproducibility.',
        119: '### Conclusions\n\nV7 improves development AUROC to **0.7181**, with **45.14% recall** and **12.5% false positives**. Copy-move detection and foreign-content localization remain weak. More clean calibration and development validation are needed before freezing the model and evaluating the final holdout.\n\n[Experiment history](../MODEL_EVOLUTION.md) · [HiRISE dataset](https://zenodo.org/records/4002935)\n\nSaved results are retained in `outputs/` and `results/development/`. Optional fresh runs write to separate folders.'
    }
    for i, value in text.items():
        assert cells[i].cell_type == 'markdown', i
        cells[i].source = value
    cells[31].source = cells[31].source.replace('## 8. Masks, noise schedule and training objective', '### Masks and training objective').replace('\ufffd', '-')
    cells[70].source = cells[70].source.replace('## 16. Continuous, normal-only calibration', '## 7. Calibration and inference').replace('\ufffd', '-')
    for i in [92,94,96,98,100,102]:
        cells[i].source = '### ' + cells[i].source.split('saved')[0].replace('### ', '').rstrip(' \ufffd-')
    # Keep optional branches after the completed experiment; remove duplicate history and unused freeze helpers.
    order = [i for i in range(63) if i != 7] + list(range(70,105)) + [110,111,119] + list(range(105,110)) + list(range(63,70))
    nb.cells = [cells[i] for i in order]
    nb.metadata.setdefault('tcmdss', {})['structured'] = True
    return nb
