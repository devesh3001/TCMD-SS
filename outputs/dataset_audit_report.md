# TCMD-SS dataset audit

Dataset infrastructure only. No anomaly model was implemented. Normal/anomaly roles remain UNASSIGNED.

## Download and extraction

Downloaded the requested version 3.2 archive directly from the [official Zenodo record](https://zenodo.org/records/4002935).

Archive bytes: 911,235,740. MD5 verified: `236d9c627db1a5970e77a01a8c8a035a`.

Archive: `data/raw/hirise-map-proj-v3_2.zip`. Extraction: `data/raw/hirise_v3_2/`.
The archive packaging folder was removed. Official label filenames contain `map-proj_v3_2` and are retained unchanged.

## Inventory and classes

129,902 physical files inspected: 64,947 valid images, 3 label/class-map files, and 64,952 AppleDouble sidecars. No corrupt or unsupported files.
Every image is a 227 x 227, one-channel grayscale JPEG. All 64,947 primary labels reconcile with images; no missing labels or files.

| ID | Class | Images | Originals | Augmented |
|---:|---|---:|---:|---:|
| 0 | other | 52,722 | 8,802 | 43,920 |
| 1 | crater | 5,024 | 794 | 4,230 |
| 2 | dark dune | 766 | 166 | 600 |
| 3 | slope streak | 1,575 | 267 | 1,308 |
| 4 | bright dune | 1,654 | 250 | 1,404 |
| 5 | impact ejecta | 476 | 74 | 402 |
| 6 | swiss cheese | 1,834 | 298 | 1,536 |
| 7 | spider | 896 | 164 | 732 |

The `other` class contains 81.18% of images; the largest class is 110.76 times the smallest. Semantic classes are not anomaly labels.

## Augmentation and supplied splits

Recovered all 10,815 base landmarks across 232 source observations. 9,022 bases have one original plus six variants; 1,793 bases have only an original.
Observed suffixes: `-r90`, `-r180`, `-r270`, `-fh`, `-fv`, `-brt`. Each occurs 9,022 times, totaling 54,132 augmented images. No base group lacks an original.

Supplied split counts below separate unique physical images from upsampling rows.

| Class | Train unique / rows | Val unique / rows | Test unique / rows |
|---|---:|---:|---:|
| other | 40,019 / 40,019 | 11,221 / 11,221 | 1,482 / 1,482 |
| crater | 3,598 / 4,333 | 1,337 / 1,435 | 89 / 89 |
| dark dune | 616 / 644 | 84 / 147 | 66 / 66 |
| slope streak | 924 / 938 | 602 / 616 | 49 / 49 |
| bright dune | 1,561 / 1,561 | 77 / 77 | 16 / 16 |
| impact ejecta | 343 / 539 | 126 / 231 | 7 / 7 |
| swiss cheese | 1,281 / 1,918 | 511 / 812 | 42 / 42 |
| spider | 637 / 1,106 | 217 / 420 | 42 / 42 |
| **Total** | **48,979 / 51,058** | **14,175 / 14,959** | **1,793 / 1,793** |

There are 2,863 repeated supplied rows. No filename, named base family or source observation spans supplied splits. The supplied allocation was inspected, not reused.

## Duplicate and leakage findings

- SHA-256 byte duplicate groups: 0. Decoded RGB pixel duplicate groups: 0.
- pHash candidate pairs at Hamming distance <= 4: 8,782; 8,640 share a base ID and 142 connect different bases.
- 24 candidate pairs cross supplied splits, creating 6 potentially leaking connected components.
- 10 candidate-linked components contain different semantic classes (261 images flagged). This is not proof of incorrect labels.
- Visual inspection of one representative pair per leaking component shows different landmarks can match because of shape or black image borders. Treat these as candidates, not confirmed same-image duplicates.
- Review image: [leakage_review.png](leakage_review.png). Full pair lists and paths are in the JSON.

The conservative graph has 10,770 landmark/duplicate components. Adding complete source-observation constraints produces 208 splitting components. No candidate links were silently discarded.

## Recommended experimental protocol

Normal classes must be selected explicitly before an unsupervised experiment. Do not interpret the provisional training partition as normal-only.
The fixed search starts at seed 42 and accepts its first class-covered group allocation (seed 44). It uses no model outcomes. Targets are 70/15/15 by group; image proportions differ.

| Provisional split | Images | Originals for evaluation |
|---|---:|---:|
| train | 39,997 | 6,829 |
| val | 14,845 | 2,377 |
| test | 10,105 | 1,609 |

Every class is represented in each proposed split. All base, duplicate, source and splitting-group overlap counts are zero; every pHash candidate pair remains together.
Class balance remains uneven, including only 16 bright-dune images in test. Revisit suitability after the normal-class definition is chosen.
`unsupervised_split` requires explicit normal class IDs and moves entire unsafe training groups to test. It raises if no normal training groups remain. This role-specific function has not been activated.
Evaluate held-out originals to avoid counting augmented variants as independent evidence. Do not infer training contamination while anomaly roles are unassigned.

## Validation and commands

18 tests passed. The real saved manifest also passed complete label matching, class-coverage, group-boundary and near-pair colocation assertions.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
curl.exe -L --fail --retry 5 --retry-all-errors --connect-timeout 30 --speed-limit 1024 --speed-time 30 -C - -o data/raw/hirise-map-proj-v3_2.zip "https://zenodo.org/records/4002935/files/hirise-map-proj-v3_2.zip?download=1"
.\.venv\Scripts\python scripts/extract_dataset.py
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -u scripts/audit_dataset.py
.\.venv\Scripts\python -m compileall -q src scripts tests
.\.venv\Scripts\python -m pip freeze
```

The initial network-restricted install and stalled download were retried successfully. Test scratch-directory permissions were resolved with project-local scratch storage. The full audit was rerun after class-coverage refinement.

## Exact project and output files created

Generated dependency/cache files are excluded from this list. Every extracted dataset file is listed in dataset_audit.csv.

```text
.gitignore
README.md
requirements.txt
requirements-lock.txt
pytest.ini
configs/base.yaml
src/__init__.py
src/data/__init__.py
src/data/dataset.py
src/data/hirise.py
src/data/split.py
src/data/transforms.py
src/utils/__init__.py
src/utils/seed.py
scripts/audit_dataset.py
scripts/extract_dataset.py
tests/test_dataset.py
outputs/dataset_audit.json
outputs/dataset_audit.csv
outputs/class_distribution.csv
outputs/base_image_groups.csv
outputs/download_verification.json
outputs/validation.json
outputs/leakage_review.png
outputs/dataset_audit_report.md
data/raw/hirise-map-proj-v3_2.zip
data/raw/hirise_v3_2/labels-map-proj_v3_2.txt
data/raw/hirise_v3_2/labels-map-proj_v3_2_train_val_test.txt
data/raw/hirise_v3_2/landmarks_map-proj-v3_2_classmap.csv
data/raw/hirise_v3_2/map-proj-v3_2/ (64,947 image files; exact names in inventory)
data/raw/hirise_v3_2/__MACOSX/ (64,952 auxiliary files; exact names in inventory)
```
