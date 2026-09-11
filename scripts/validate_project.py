"""Validate saved experimental artifacts without rerunning or tuning a model."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import nbformat
from src.utils.config import ROOT,load_config,output_dir,save_json
from src.data.protocol import assert_normal,assert_disjoint
from src.fusion.calibration import BRANCHES,VARIANTS,fuse
from src.pipeline import model_hashes
from src.evaluation.metrics import metrics


def validate(config):
    out=output_dir(config)
    read=lambda path:json.loads((out/path).read_text())
    parts={name:pd.read_csv(ROOT/f'data/splits/{name}.csv',low_memory=False) for name in ['train_normal','calibration_normal','test_clean']}
    assert_disjoint(parts)
    for frame in parts.values():assert_normal(frame)
    assert parts['calibration_normal'].is_original.all() and parts['test_clean'].is_original.all()
    normal=pd.read_csv(out/'calibration/normal_scores.csv',low_memory=False);assert_normal(normal)
    training=pd.read_csv(out/'diffusion/training_manifest.csv',low_memory=False);assert_normal(training)
    benchmark=pd.read_csv(out/'benchmark/manifest.csv',low_memory=False)
    assert set(benchmark.base_id).issubset(set(parts['test_clean'].base_id))
    assert benchmark.is_anomaly.eq(1).all() and benchmark.changed_fraction.gt(0).all()
    assert_disjoint({'train':training,'calibration':normal,'test':benchmark})
    state=read('calibration/calibration.json')
    assert state['model_hashes']==model_hashes(config) and state['config']==config
    assert state['weights']==VARIANTS
    for variant in VARIANTS:
        assert np.isclose(np.quantile(fuse(normal[BRANCHES],state,variant),.99),state['thresholds'][variant])
    predictions=pd.read_csv(out/'evaluation/predictions.csv',low_memory=False)
    assert np.isfinite(predictions[BRANCHES+['score']].to_numpy()).all()
    assert np.allclose(fuse(predictions[BRANCHES],state),predictions.score)
    measured=metrics(predictions.is_anomaly,predictions.score,state['thresholds']['V3_TCMD_SS'])
    stored=read('evaluation/metrics.json')
    for key,value in measured.items():
        assert value is None and stored[key] is None or np.isclose(value,stored[key])
    ablations=pd.read_csv(out/'ablation_results.csv')
    assert set(ablations.variant)==set(VARIANTS)
    for row in ablations.to_dict('records'):
        score=fuse(predictions[BRANCHES],state,row['variant'])
        assert np.isclose(metrics(predictions.is_anomaly,score,row['threshold'])['auroc'],row['auroc'])
    notebook=nbformat.read(ROOT/'TCMD_SS_HiRISE_Anomaly_Detection.ipynb',as_version=4)
    code=[cell for cell in notebook.cells if cell.cell_type=='code']
    assert all(cell.execution_count is not None for cell in code)
    assert any(cell.outputs for cell in code)
    assert not any(output.output_type=='error' for cell in code for output in cell.outputs)
    assert (ROOT/'report/TCMD_SS_Report.pdf').stat().st_size>1000
    assert (ROOT/'MODEL_EVOLUTION.md').exists()
    for example in read('heatmaps/explanations.json'):
        assert (ROOT/example['figure']).exists()
    with (ROOT/'outputs/dataset_audit.csv').open('rb') as stream: audit_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    protocol=json.loads((ROOT/'outputs/splits/protocol.json').read_text())
    assert protocol['audit_sha256']==audit_hash
    checks={'audit_preserved':True,'forbidden_group_overlap':0,'training_and_calibration_clean_only':True,
            'benchmark_test_only':True,'original_calibration_and_test':True,'normal_only_threshold_verified':True,
            'four_branch_scores_finite':True,'saved_metrics_recomputed':True,'all_ablations_verified':True,
            'notebook_code_cells_executed':len(code),'pdf_exists':True,'evolution_exists':True,
            'training_pool':len(training),'calibration_count':len(normal),'test_count':len(predictions),
            'configuration':config,'provisional':True}
    save_json(out/'final_validation.json',checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/local.yaml');args=parser.parse_args()
    validate(load_config(args.config))
