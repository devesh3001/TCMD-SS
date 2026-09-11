"""Run a bounded three-channel generative development ablation without DINO.

This is deliberately a named partial model, never a silent replacement for the
four-channel candidate. Cached reconstructions can later support DINO evaluation.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from scripts.prepare_revision import read, disjoint, preserve_write
from scripts.evaluate_cached_fusion import metrics, write_csv
from src.diffusion.model import MaskedUNet
from src.generative.discrepancy import normal_reconstruction, local_energy, directional_features, top_fraction
from src.generative.stress import FAMILIES, case_seed, stress, tensor_image, pil_image


def choose(rows, count):
    groups={}
    for r in rows:groups.setdefault(r['class_id'],[]).append(r)
    for key in groups:
        groups[key]=sorted(groups[key],key=lambda r:hashlib.sha256(('42'+r['base_id']).encode()).hexdigest())
    result=[]
    while len(result)<count and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(result)<count: result.append(groups[key].pop())
    return result


def score_maps(observed, generated, uncertainty):
    pixel=(observed-generated).abs()/(1+4*uncertainty)
    texture=(torch.log(local_energy(generated)+1e-4)-torch.log(local_energy(observed)+1e-4)).clamp_min(0)
    spectral=(directional_features(observed)-directional_features(generated)).abs()
    rows=(observed.mean(-1)-generated.mean(-1)).abs()[...,None]
    cols=(observed.mean(-2)-generated.mean(-2)).abs()[:,:,None,:]
    maps=torch.cat([pixel,texture,(rows+cols)/2],dim=1)
    scores=torch.stack([top_fraction(pixel),top_fraction(texture),top_fraction(spectral)],dim=1)
    return scores,maps


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--calibration',type=int,default=64);parser.add_argument('--development',type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    directory=ROOT/'results/development/generative_v6'
    cache=ROOT/'outputs/generative_v6_cache';cache.mkdir(parents=True,exist_ok=True)
    parts={n:read(ROOT/f'data/splits/generative/{n}.csv') for n in ['train_normal','calibration_a','calibration_b','development','final_holdout']}
    disjoint(parts)
    consumed={r['base_id'] for r in read(ROOT/'outputs/evaluation/predictions.csv') if float(r['is_anomaly'])==0}
    selected={n:choose(parts[n],args.calibration) for n in ['calibration_a','calibration_b']}
    selected['development']=choose([r for r in parts['development'] if r['base_id'] in consumed],args.development)
    disjoint(selected)
    checkpoint=ROOT/'outputs/diffusion/checkpoints/best.pt'
    files=[checkpoint,Path(__file__),ROOT/'src/generative/stress.py',ROOT/'src/generative/discrepancy.py',ROOT/'src/diffusion/process.py',ROOT/'src/diffusion/model.py']
    config={'method':'generative pixel/texture/directional spectral; NO semantic branch', 'seed':42,'seeds':[42,1042],
            'sampling_steps':15,'diffusion_steps':100,'size':128,'top_fraction':.005,'calibration':args.calibration,'development':args.development,
            'device':device,'torch':torch.__version__,'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'bases':{n:[r['base_id'] for r in rows] for n,rows in selected.items()}}
    preserve_write(directory/'run_config.json',json.dumps(config,indent=2)+'\n')
    if (directory/'summary.json').exists():
        print((directory/'summary.json').read_text());return
    for name,rows in selected.items():write_csv(directory/f'{name}_manifest.csv',rows)
    jobs=[]
    for role,rows in selected.items():
        for row in rows:
            jobs.append({**row,'role':role,'family':'clean','severity':0.,'case_id':row['base_id']+'_clean'})
            if role=='development':
                for family in FAMILIES:
                    for severity in [.15,.30,.50]:
                        jobs.append({**row,'role':role,'family':family,'severity':severity,'is_anomaly':1,
                                     'origin':'development_corruption','case_id':f"{row['base_id']}_{family}_{severity}"})
    write_csv(directory/'jobs.csv',jobs)
    model=MaskedUNet(32).to(device)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
    model.eval()
    pending=[r for r in jobs if not (cache/f"{r['case_id']}.pt").exists()]
    start=time.perf_counter();inference_seconds=0.
    for offset in range(0,len(pending),4):
        batch=pending[offset:offset+4];inputs=[];masks=[]
        for row in batch:
            with Image.open(row['image_path']) as original:
                image=original.convert('L').resize((128,128),Image.Resampling.BILINEAR)
            mask=Image.new('L',(128,128))
            if row['family']!='clean':image,mask=stress(image,row['family'],row['severity'],case_seed(row['base_id'],row['family']))
            inputs.append(tensor_image(image));masks.append(tensor_image(mask))
        x=torch.stack(inputs).to(device)
        if device=='cuda':torch.cuda.synchronize()
        before=time.perf_counter()
        with torch.no_grad():
            generated,uncertainty=normal_reconstruction(model,x)
            scores,maps=score_maps(x,generated,uncertainty)
        if device=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-before;inference_seconds+=elapsed
        if not torch.isfinite(scores).all() or not torch.isfinite(maps).all():raise ValueError('Nonfinite inference')
        for i,row in enumerate(batch):
            torch.save({'case_id':row['case_id'],'scores':scores[i].cpu(),'maps':maps[i].cpu(),'input':x[i].cpu(),
                        'reconstruction':generated[i].cpu(),'uncertainty':uncertainty[i].cpu(),'mask':masks[i],
                        'seconds':elapsed/len(batch)},cache/f"{row['case_id']}.pt")
        print(f"Cached {min(offset+4,len(pending))}/{len(pending)} new cases; {time.perf_counter()-start:.1f}s",flush=True)
    records=[];a=[];b=[];dev=[]
    for row in jobs:
        item=torch.load(cache/f"{row['case_id']}.pt",weights_only=True)
        values=item['scores'].tolist()
        record={**row,'pixel':values[0],'texture':values[1],'spectral':values[2],'seconds':item['seconds']}
        records.append(record)
        {'calibration_a':a,'calibration_b':b,'development':dev}[row['role']].append(record)
    write_csv(directory/'raw_scores.csv',records)
    med=torch.tensor([statistics.median(r[k] for r in a) for k in ['pixel','texture','spectral']])
    scale=torch.tensor([max(1e-6,1.4826*statistics.median(abs(r[k]-float(med[i])) for r in a)) for i,k in enumerate(['pixel','texture','spectral'])])
    def normalise(rows):return (torch.tensor([[r[k] for k in ['pixel','texture','spectral']] for r in rows])-med)/scale
    zb,zd=normalise(b),normalise(dev)
    variants={'pixel':[0],'texture':[1],'spectral':[2],'pixel_texture':[0,1],'pixel_spectral':[0,2],'complete_three':[0,1,2]}
    results=[];predictions=[];details=[]
    for name,indices in variants.items():
        for agg in ['mean','max'] if len(indices)>1 else ['mean']:
            def combine(z):return z[:,indices].mean(1) if agg=='mean' else z[:,indices].max(1).values
            threshold=float(torch.quantile(combine(zb),.95));scores=combine(zd).tolist();variant=f'{name}_{agg}'
            results.append({'variant':variant,**metrics(dev,scores,threshold)})
            for row,value in zip(dev,scores):predictions.append({'variant':variant,'base_id':row['base_id'],'source_id':row['source_id'],'case_id':row['case_id'],'family':row['family'],'severity':row['severity'],'is_anomaly':int(float(row['is_anomaly'])),'score':value,'threshold':threshold})
            for dimension in ['family','severity']:
                for value in sorted({r[dimension] for r in dev if int(float(r['is_anomaly']))}):
                    select=[i for i,r in enumerate(dev) if not int(float(r['is_anomaly'])) or r[dimension]==value]
                    details.append({'variant':variant,'dimension':dimension,'value':value,**metrics([dev[i] for i in select],[scores[i] for i in select],threshold)})
    write_csv(directory/'ablations.csv',results);write_csv(directory/'predictions.csv',predictions);write_csv(directory/'by_family_severity.csv',details)
    state={'status':'bounded_development_only','normal_calibration_a':len(a),'normal_calibration_b':len(b),'development_cases':len(dev),
           'median':med.tolist(),'scale':scale.tolist(),'threshold_procedure':'clean A median/MAD; independent clean B q95',
           'best_diagnostic':max(results,key=lambda r:r['auroc']),'complete_three_max':next(r for r in results if r['variant']=='complete_three_max'),
           'mean_inference_seconds':statistics.mean(r['seconds'] for r in records),'parameters':sum(p.numel() for p in model.parameters()),
           'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,'final_evaluated':False,
           'limitations':['No DINO semantic branch: this is a partial-model ablation','Six synthetic families only; foreign content is procedural, not a downloaded dataset','16 development originals; dependent corruptions are not independent samples','No model changes or threshold fitting from anomaly labels','No architecture freeze or final evaluation']}
    preserve_write(directory/'summary.json',json.dumps(state,indent=2)+'\n');print(json.dumps(state,indent=2))


if __name__=='__main__':main()
