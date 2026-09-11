"""Analytical prevalence sensitivity; no new test images or fitted thresholds."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_cached_fusion import read, metrics, curves, write_csv
from scripts.prepare_revision import preserve_write


def prevalence_metrics(labels,scores,threshold,prevalence):
    curve=curves(labels,scores)
    weighted_ap=0.
    for a,b in zip(curve,curve[1:]):
        denominator=prevalence*b['tpr']+(1-prevalence)*b['fpr']
        precision=prevalence*b['tpr']/denominator if denominator else 1.
        weighted_ap+=(b['recall']-a['recall'])*precision
    tp=sum(y==1 and s>threshold for y,s in zip(labels,scores))/sum(labels)
    fp=sum(y==0 and s>threshold for y,s in zip(labels,scores))/(len(labels)-sum(labels))
    denominator=prevalence*tp+(1-prevalence)*fp
    precision=prevalence*tp/denominator if denominator else 0.
    return {'assumed_anomaly_prevalence':prevalence,'tpr':tp,'fpr':fp,'expected_precision':precision,
            'reweighted_average_precision':weighted_ap,'expected_false_alerts_per_1000':1000*(1-prevalence)*fp,
            'expected_true_alerts_per_1000':1000*prevalence*tp}


def main():
    rows=read(ROOT/'outputs/evaluation/predictions.csv')
    cal=json.loads((ROOT/'outputs/calibration/calibration.json').read_text())
    labels=[int(float(r['is_anomaly'])) for r in rows]
    variants={'fixed_baseline':[float(r['score']) for r in rows]}
    variants['historical_equal_weight']=[sum(max(0,min(100,(float(r[k])-m)/s))*.25 for k,m,s in zip(cal['branches'],cal['median'],cal['scale'])) for r in rows]
    thresholds={'fixed_baseline':cal['thresholds']['V3_TCMD_SS'],'historical_equal_weight':cal['thresholds']['equal_weight']}
    output=[]
    for name,scores in variants.items():
        for prevalence in [.01,.05,.1]:
            output.append({'variant':name,**prevalence_metrics(labels,scores,thresholds[name],prevalence)})
    directory=ROOT/'results/development/rare_prevalence'
    write_csv(directory/'sensitivity.csv',output)
    note={'method':'Analytical class-prevalence reweighting of historical development conditional score distributions',
          'reference':'VAD4Space, arXiv:2603.13993, evaluation setting with 5% anomalies',
          'not_a_new_benchmark':True,'thresholds_changed':False,'external_data_used':False,
          'limitation':'Assumes class-conditional score distributions remain unchanged; synthetic prevalence calculations are not deployment validation.'}
    preserve_write(directory/'method.json',json.dumps(note,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
