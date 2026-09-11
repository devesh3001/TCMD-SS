"""Development V7: test localized residual and narrow-band discrepancy hypotheses."""
import json
import statistics
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from scripts.evaluate_cached_fusion import read, write_csv, metrics
from scripts.prepare_revision import preserve_write
from src.generative.discrepancy import top_fraction


def frequency_peaks(images):
    profiles=torch.stack([images.mean(-1),images.mean(-2)],dim=2).squeeze(1)
    profiles=profiles-profiles.mean(-1,keepdim=True)
    power=torch.fft.rfft(profiles*torch.hann_window(128),norm='ortho').abs().square()
    # Nearby frequencies estimate broadband terrain energy; exclude the queried bin.
    background=(F.avg_pool1d(power,9,stride=1,padding=4)*9-power)/8
    return torch.log1p(power/(background+1e-4))[...,3:]


def robust_scale(values):
    median=torch.quantile(values,.5,dim=0)
    mad=1.4826*torch.quantile((values-median).abs(),.5,dim=0)
    # A spike at zero makes MAD degenerate; normal-only spread supplies a finite fallback.
    spread=(torch.quantile(values,.9,dim=0)-torch.quantile(values,.1,dim=0))/2.563
    return median,torch.where(mad>1e-6,mad,spread).clamp_min(1e-6)


def refined(item):
    x=item['input'][None];g=item['reconstruction'][None]
    residual=item['maps'][0:1][None]
    local=F.avg_pool2d(residual,15,stride=1,padding=7)
    # Suppress broad reconstruction mismatch while retaining isolated excess residual.
    pixel=(residual/(local+.03))[...,4:-4,4:-4]
    spectral=(frequency_peaks(x)-frequency_peaks(g)).clamp_min(0)
    texture=item['maps'][1][None,None,...]
    return torch.stack([top_fraction(pixel),top_fraction(texture),top_fraction(spectral)],dim=1)[0]


def main():
    torch.set_num_threads(4)
    source=ROOT/'results/development/generative_v6';target=ROOT/'results/development/generative_v7'
    rows=read(source/'raw_scores.csv');cache=ROOT/'outputs/generative_v6_cache'
    scores=torch.stack([refined(torch.load(cache/f"{r['case_id']}.pt",weights_only=True)) for r in rows])
    ai=[i for i,r in enumerate(rows) if r['role']=='calibration_a'];bi=[i for i,r in enumerate(rows) if r['role']=='calibration_b'];di=[i for i,r in enumerate(rows) if r['role']=='development']
    med,scale=robust_scale(scores[ai]);zb=(scores[bi]-med)/scale;zd=(scores[di]-med)/scale
    dev=[rows[i] for i in di]
    variants={'localized_pixel':[0],'texture':[1],'narrowband_spectral':[2],'pixel_spectral':[0,2],'complete_three':[0,1,2]}
    output=[];details=[];predictions=[]
    for name,indices in variants.items():
        for aggregation in ['mean','max'] if len(indices)>1 else ['mean']:
            def aggregate(z):return z[:,indices].mean(1) if aggregation=='mean' else z[:,indices].max(1).values
            threshold=float(torch.quantile(aggregate(zb),.95));values=aggregate(zd).tolist();variant=f'{name}_{aggregation}'
            output.append({'variant':variant,**metrics(dev,values,threshold)})
            for r,s in zip(dev,values):predictions.append({'variant':variant,'base_id':r['base_id'],'source_id':r['source_id'],'case_id':r['case_id'],'is_anomaly':r['is_anomaly'],'family':r['family'],'severity':r['severity'],'score':s,'threshold':threshold})
            for dimension in ['family','severity']:
                for value in sorted({r[dimension] for r in dev if int(float(r['is_anomaly']))}):
                    take=[i for i,r in enumerate(dev) if not int(float(r['is_anomaly'])) or r[dimension]==value]
                    details.append({'variant':variant,'dimension':dimension,'value':value,**metrics([dev[i] for i in take],[values[i] for i in take],threshold)})
    write_csv(target/'ablations.csv',output);write_csv(target/'predictions.csv',predictions);write_csv(target/'by_family_severity.csv',details)
    summary={'status':'development_only','method':'localized generative residual / texture deficiency / narrowband observed-generated discrepancy',
             'median':med.tolist(),'scale':scale.tolist(),'normal_calibration_a':len(ai),'normal_calibration_b':len(bi),
             'best_diagnostic':max(output,key=lambda r:r['auroc']),'complete_three_max':next(r for r in output if r['variant']=='complete_three_max'),
             'threshold_procedure':'normal A robust statistics, normal B q95','new_inference':False,'final_evaluated':False,
             'limitations':['No DINO; same 16 development originals and cached two-seed reconstructions','Scores selected using development; not untouched evidence','Residual aggregation excludes four-pixel frame edge; edge faults may be missed']}
    preserve_write(target/'summary.json',json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
