"""Create and execute the submission notebook; derive evolution notes from saved evidence."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from src.utils.config import ROOT,load_config,output_dir


def build(config_path: str, execute: bool = True):
    config=load_config(config_path);out=output_dir(config)
    ablation=pd.read_csv(out/'ablation_results.csv').set_index('variant')
    training=json.loads((out/'diffusion/training_summary.json').read_text())
    results=json.loads((out/'evaluation/metrics.json').read_text())
    semantic=json.loads((out/'semantic/metadata.json').read_text())
    v1,v2,v3=[ablation.loc[key] for key in ['V1_diffusion','V2_diffusion_semantic','V3_TCMD_SS']]
    evolution=f'''# MODEL EVOLUTION

These are three implemented nested versions of ONE TCMD-SS system. The local study
uses one bounded diffusion checkpoint and predefined score ablations, not three
independent full training runs. Additions and fusion weights were specified before
corruption evaluation; the following evidence is post-run analysis, not benchmark tuning.

# Version 1 - Masked Diffusion Baseline

Symptom: The bounded run's best normal validation reconstruction L1 is
{training['best_normal_validation_l1']:.6f}. Diffusion-only evaluation has
{int(v1['fp'])} false positives and {int(v1['fn'])} false negatives at its normal-only
threshold; AUROC is {v1['auroc']:.6f}.

Diagnosis: The actual training history and saved reconstruction panels document a
short {training['epochs_completed']}-epoch run. Pixel reconstruction alone is not a
complete terrain representation. This experiment does not prove that longer
training could not resolve some of its errors.

Fix: V2 adds frozen, pretrained DINO patch-memory distances for semantic context.
No corruption is added to diffusion training or the normal memory bank.

# Version 2 - Diffusion + Semantic Context

Symptom: V2 produces {int(v2['fp'])} false positives and {int(v2['fn'])} false negatives;
AUROC is {v2['auroc']:.6f}. It does not eliminate detection errors.

Diagnosis: The actual frozen backbone is {semantic['actual_model']}. The CPU smoke used
ImageNet DINO-small; the bounded GPU run's backbone is recorded explicitly.
Domain-transfer quality for real faults is not established by synthetic metrics. Dense feature and exact chunked-kNN tests
verify the implemented feature path independently of benchmark outcomes.

Fix: The predefined V3 system adds local multi-scale frequency statistics and a
PCA/shrinkage latent distance, both fit using normal training data only.

# Version 3 - TCMD-SS

Symptom: V3 still has {int(v3['fp'])} false positives and {int(v3['fn'])} false negatives;
AUROC is {v3['auroc']:.6f}. Small normal calibration samples make the 99th percentile
uncertain. Class-specific false-positive counts are saved separately.

Diagnosis: Actual engineering tests exposed an empty FFT band for 8-pixel windows
and a Windows-blocked scikit-learn covariance DLL. These were implementation/runtime
issues, not evidence of a scientific advantage. Near-zero MAD also needs an explicit
numerical floor in normal-only calibration.

Fix: Frequency bands now have support at every configured scale; NumPy PCA with
fixed covariance shrinkage avoids the blocked DLL. Robust normal-only normalization,
fixed fusion and independently normal-calibrated ablation thresholds are implemented.
All corrections were applied before final benchmark evaluation. No weights were
changed to improve anomaly AUROC. Remaining full-training work is documented in README.

Evidence: outputs/diffusion/training_history.csv, outputs/evaluation/metrics.json,
outputs/evaluation/normal_fpr_by_class.csv, outputs/ablation_results.csv and outputs/heatmaps/.
'''
    (ROOT/'MODEL_EVOLUTION.md').write_text(evolution,encoding='utf-8')
    nb=nbf.v4.new_notebook();cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):cells.append(nbf.v4.new_code_cell(text))
    md('# TCMD-SS: HiRISE structural anomaly detection\n\n**Provisional, real bounded experiment.** Terrain-Conditioned Masked Diffusion with Semantic-Spectral Scoring. All outputs below are executed from saved artifacts, not invented results.')
    md('## 1. Problem definition\nLearn genuine Martian imagery without anomaly labels, then flag structural deviations such as acquisition artifacts, crop corruption and foreign visual patterns.')
    code(f'''from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd()
if not (ROOT / 'src').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.utils.config import load_config, output_dir
config = load_config({config_path!r})
out = output_dir(config)
read = lambda name: json.loads((out / name).read_text())
display(read('run_summary.json'))''')
    md('## 2. Semantic classes are all normal\nHiRISE class IDs 0-7 describe legitimate terrain. Craters, dunes, streaks, ejecta, swiss cheese and spiders are not anomaly labels. Labels support coverage diagnostics only; no anomaly labels enter fitting, calibration or threshold selection.')
    md('## 3. Dataset overview and existing audit')
    code("display(pd.read_csv(ROOT/'outputs/class_distribution.csv'))\ndisplay(json.loads((ROOT/'outputs/download_verification.json').read_text()))")
    md('## 4. Leakage-safe protocol\nThe existing audited allocation is preserved, including base IDs, source observations and duplicate/pHash connected components. Calibration and test are originals. Online masks are conditions, not labeled anomalies.')
    code("from src.data.protocol import assert_disjoint\nparts={name:pd.read_csv(ROOT/f'data/splits/{name}.csv',low_memory=False) for name in ['train_normal','calibration_normal','test_clean']}\nassert_disjoint(parts)\ndisplay({key:len(value) for key,value in parts.items()})")
    md('## 5. TCMD-SS architecture\n```text\nQuery grayscale image\n  +-- masked conditional diffusion --> reconstruction residual\n  +-- frozen DINO patch memory ------> semantic map\n  +-- multi-scale FFT statistics ----> spectral map\n  +-- pooled PCA/shrinkage ----------> latent score\n              |\n     normal-only robust calibration\n              |\n       fixed 0.40/0.25/0.20/0.15 fusion\n              |\n       image score + heatmap + explanation\n```')
    md('## 6. Masked diffusion\nThe U-Net predicts noise in 10-40% hidden regions, conditioned only on visible terrain. Four complementary inference masks cover every pixel. Eta=0 DDIM yields reconstruction residuals. Checkpoint selection uses normal validation sources drawn from the training partition. Set RUN_TRAINING to True only when intentionally continuing training; increase config epochs before resume.')
    code("from src.diffusion.train import train\nRUN_TRAINING = False\nif RUN_TRAINING:\n    train(config, resume=str(out/'diffusion/checkpoints/last.pt'))\ndisplay(pd.read_csv(out/'diffusion/training_history.csv'))")
    md('## 7. Frozen semantic branch\nGrayscale is repeated into RGB without inventing color. Pretrained dense patch tokens are L2-normalized. The normal-only memory uses balanced base sampling and chunked k=5 cosine distance. The top 5% patch distances define the semantic image score.')
    code("display(read('semantic/metadata.json'))")
    md('## 8. Spectral branch\nLocal 8/16/32-pixel FFT radial-band and directional-energy features are compared with robust normal statistics. No additional deep network or anomaly fitting is used.')
    md('## 9. Latent branch\nPooled normal DINO embeddings are reduced by NumPy PCA, followed by fixed isotropic covariance shrinkage and Mahalanobis distance. The latent contribution is image-global, not localized.')
    md('## 10. Normal-only calibration\nPositive robust z scores use median and 1.4826*MAD + epsilon. Fusion weights were fixed before evaluation. The detection threshold is the final-score 99th percentile on calibration normals only.')
    code("calibration=read('calibration/calibration.json')\ndisplay({k:calibration[k] for k in ['normal_count','median','scale','thresholds','weights']})")
    md('## 11. Evaluation protocol\nOnly held-out test originals seed the corruption benchmark. Multiple controlled severities cover acquisition, structural and synthetic foreign-content families. No benchmark result is used to retrain or tune weights. AUROC confidence intervals bootstrap whole base landmarks.')
    code("benchmark=pd.read_csv(out/'benchmark/manifest.csv',low_memory=False)\nassert set(benchmark.base_id).issubset(set(parts['test_clean'].base_id))\ndisplay(benchmark.groupby(['family','severity']).size().rename('count').to_frame())")
    md('## 12. Actual results\nThese are provisional controlled-corruption measurements, not real sensor-failure validation. AUPRC depends on the benchmark prevalence.')
    code("display(read('evaluation/metrics.json'))\ndisplay(pd.read_csv(out/'evaluation/by_family.csv'))\ndisplay(pd.read_csv(out/'evaluation/by_severity.csv'))\ndisplay(pd.read_csv(out/'evaluation/normal_fpr_by_class.csv'))")
    md('## 13. Heatmaps and explanations\nEach figure shows input, reconstruction, residual, semantic and spectral maps, fused map, overlay and actual weighted contributions. These are explanatory diagnostics, not causal or segmentation guarantees.')
    code("examples=read('heatmaps/explanations.json')\nfor example in [examples[0],examples[-1]]:\n    display(Markdown(example['explanation']))\n    display(Image(filename=str(ROOT/example['figure'])))")
    md('## 14. Predefined ablations\nV1, V2 and V3 share a trained checkpoint and differ in scoring components. They are not three independent full training runs. Each threshold is calibrated on normals separately. Do not infer superiority without evidence.')
    code("display(pd.read_csv(out/'ablation_results.csv'))\ndisplay(Image(filename=str(out/'ablation_plot.png')))")
    md('## 15. Failure cases\nInspect actual high-scoring clean examples and missed test corruptions. A rare semantic class is not automatically an anomaly.')
    code("predictions=pd.read_csv(out/'evaluation/predictions.csv',low_memory=False)\ncolumns=['base_id','class_name','family','severity','score','flagged']\ndisplay(predictions[predictions.is_anomaly.eq(0)].nlargest(5,'score')[columns])\ndisplay(predictions[predictions.is_anomaly.eq(1)&~predictions.flagged].nsmallest(5,'score')[columns])")
    md('## 16. Conclusions and reproducibility\nThe four-branch generative pipeline runs end to end with normal-only calibration. Short training, small calibration/test subsets, synthetic-only faults and source dependence limit claims. See README for full/resumed training, MODEL_EVOLUTION.md for actual evidence, and report/TCMD_SS_Report.pdf for the generated report.')
    nb.cells=cells
    nb.metadata['kernelspec']={'display_name':'TCMD-SS Python','language':'python','name':'tcmdss'}
    path=ROOT/'TCMD_SS_HiRISE_Anomaly_Detection.ipynb'
    nbf.write(nb,path)
    if execute:
        kernel_dir=out/'jupyter/kernels/tcmdss';kernel_dir.mkdir(parents=True,exist_ok=True)
        (kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'TCMD-SS Python','language':'python'}))
        runtime=out/'jupyter/runtime';runtime.mkdir(parents=True,exist_ok=True)
        os.environ['JUPYTER_RUNTIME_DIR']=str(runtime)
        os.environ['IPYTHONDIR']=str(out/'jupyter/ipython')
        manager=KernelManager(kernel_name='tcmdss',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
        client=NotebookClient(nb,timeout=300,km=manager,resources={'metadata':{'path':str(ROOT)}})
        client.execute();nbf.write(nb,path)
    print(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/local.yaml');parser.add_argument('--no-execute',action='store_true')
    args=parser.parse_args();build(args.config,not args.no_execute)
