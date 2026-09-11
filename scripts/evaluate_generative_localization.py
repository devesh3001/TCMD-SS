"""Clean-only pixel-map calibration and source-bootstrap development diagnostics."""
import json
import random
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from scripts.evaluate_cached_fusion import read, metrics, write_csv, quantile
from scripts.prepare_revision import preserve_write


def pixel_auc(mask,scores):
    mask=mask.flatten().long();scores=scores.flatten()
    positives=int(mask.sum());negatives=len(mask)-positives
    if not positives or not negatives:return None
    order=scores.argsort();values=scores[order];labels=mask[order]
    _,inverse,counts=torch.unique_consecutive(values,return_inverse=True,return_counts=True)
    positive_counts=torch.zeros(len(counts),dtype=torch.float64).scatter_add_(0,inverse,labels.double())
    negative_counts=counts.double()-positive_counts
    lower_negatives=negative_counts.cumsum(0)-negative_counts
    return float((positive_counts*(lower_negatives+.5*negative_counts)).sum()/(positives*negatives))


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v6','v7'],default='v6')
    args=parser.parse_args()
    torch.set_num_threads(4)
    directory=ROOT/f'results/development/generative_{args.version}';cache=ROOT/'outputs/generative_v6_cache'
    if not (directory/'summary.json').exists():raise RuntimeError('Wait for development inference to complete')
    rows=read(ROOT/'results/development/generative_v6/raw_scores.csv')
    def load(row):
        item=torch.load(cache/f"{row['case_id']}.pt",weights_only=True)
        if args.version=='v7':
            residual=item['maps'][0:1][None]
            item['maps'][0]=(residual/(F.avg_pool2d(residual,15,stride=1,padding=7)+.03))[0,0]
            item['maps'][:,:4]=0;item['maps'][:,-4:]=0;item['maps'][:,:,:4]=0;item['maps'][:,:,-4:]=0
        return item
    # Uniform per-image spatial subsampling limits memory and avoids area-rich images dominating.
    a=torch.cat([load(r)['maps'].flatten(1)[:,::16] for r in rows if r['role']=='calibration_a'],dim=1)
    median=torch.quantile(a,.5,dim=1);scale=(1.4826*torch.quantile((a-median[:,None]).abs(),.5,dim=1)).clamp_min(1e-6)
    if args.version=='v7':
        from scripts.refine_generative_scores import robust_scale
        median,scale=robust_scale(a.T)
    def map_score(maps):
        z=(maps-median[:,None,None])/scale[:,None,None]
        return z.mean(0) if args.version=='v7' else z.max(0).values
    b=torch.cat([map_score(load(r)['maps']).flatten()[::16] for r in rows if r['role']=='calibration_b'])
    threshold=float(torch.quantile(b,.95))
    output=[]
    for r in rows:
        if r['role']!='development' or not int(float(r['is_anomaly'])):continue
        item=load(r);mask=item['mask'][0]>0;score=map_score(item['maps']);flag=score>threshold
        inside=score[mask];outside=score[~mask]
        # Positive-map ratio is descriptive; near-zero background can make it unstable.
        ratio=float(inside.clamp_min(0).mean()/(outside.clamp_min(0).mean()+1e-6)) if len(outside) else None
        union=int((flag|mask).sum())
        output.append({'case_id':r['case_id'],'base_id':r['base_id'],'source_id':r['source_id'],'family':r['family'],'severity':r['severity'],
                       'pixel_auroc':pixel_auc(mask,score),'iou_at_clean_pixel_q95':float((flag&mask).sum())/union if union else None,
                       'positive_map_inside_outside_ratio':ratio,'mask_fraction':float(mask.float().mean())})
    write_csv(directory/'localization.csv',output)
    by_family=[]
    for family in sorted({r['family'] for r in output}):
        part=[r for r in output if r['family']==family]
        result={'family':family,'cases':len(part)}
        for key in ['pixel_auroc','iou_at_clean_pixel_q95','positive_map_inside_outside_ratio']:
            values=[r[key] for r in part if r[key] is not None]
            result[key]=statistics.mean(values) if values else None
        by_family.append(result)
    write_csv(directory/'localization_by_family.csv',by_family)
    variant='complete_three_mean' if args.version=='v7' else 'complete_three_max'
    predictions=[r for r in read(directory/'predictions.csv') if r['variant']==variant]
    groups={}
    for row in predictions:groups.setdefault(row['source_id'],[]).append(row)
    sources=list(groups);rng=random.Random(42);aucs=[]
    for _ in range(200):
        sample=[r for _ in sources for r in groups[rng.choice(sources)]]
        if len({r['is_anomaly'] for r in sample})!=2:continue
        aucs.append(metrics(sample,[float(r['score']) for r in sample],float(sample[0]['threshold']))['auroc'])
    state={'map_method':('mean of clean-A robust z with zero-MAD spread fallback; localized pixel/texture/profile maps' if args.version=='v7' else 'maximum of clean-A pixelwise robust z for pixel/texture/profile maps'),
           'map_median':median.tolist(),'map_scale':scale.tolist(),'pixel_threshold':threshold,
           'pixel_threshold_source':'independent clean calibration B; uniform spatial subsampling; q95',
           'source_bootstrap':{'variant':variant,'sources':len(sources),'replicates':len(aucs),'auroc_lower':quantile(aucs,.025),'auroc_upper':quantile(aucs,.975)},
           'limitations':['This is a three-channel development ablation without DINO','Full-image stripe masks have no background: pixel AUROC and ratio are undefined','Frequency profile maps are row/column attribution, not spatial inverse FFT','Geometric corruption masks, not manually annotated real faults','Small source count makes bootstrap intervals unstable'],
           'final_evaluated':False}
    preserve_write(directory/'localization_summary.json',json.dumps(state,indent=2)+'\n')
    print(json.dumps(state,indent=2));print(json.dumps(by_family,indent=2))


if __name__=='__main__':main()
