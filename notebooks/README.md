# Running the notebook

Open [TCMD_SS_Final.ipynb](TCMD_SS_Final.ipynb).

1. Attach the project folder, including results, split manifests and checkpoint, plus the extracted HiRISE dataset.
2. Enable a GPU in Kaggle or Colab.
3. Set `PROJECT_ROOT_OVERRIDE` and `DATASET_ROOT_OVERRIDE` if needed.
4. Run the cells in order.

The default mode displays saved results and runs a reconstruction demo. Set `RUN_TRAINING` or `RUN_DEVELOPMENT_INFERENCE` to run new experiments. New outputs go into separate folders. `RUN_DINO` requires additional dependencies.

All 77 code cells passed locally in the default mode; optional training and full inference were not run. Previous notebook copies are in `archive/`.
