"""Render measured development maps and comparisons as labelled report figures."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from PIL import ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import pymupdf
from scripts.evaluate_cached_fusion import read
from src.generative.stress import pil_image


def main():
    torch.set_num_threads(4)
    target=ROOT/'results/figures/generative_v7';target.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/development/generative_v7'
    cache=ROOT/'outputs/generative_v6_cache'
    state=json.loads((source/'localization_summary.json').read_text())
    median=torch.tensor(state['map_median'])[:,None,None];scale=torch.tensor(state['map_scale'])[:,None,None]
    predictions=[r for r in read(source/'predictions.csv') if r['variant']=='complete_three_mean']
    figures=[]
    for family in sorted({r['family'] for r in predictions if float(r['is_anomaly'])}):
        row=next(r for r in predictions if r['family']==family and float(r['severity'])==.3)
        item=torch.load(cache/f"{row['case_id']}.pt",weights_only=True)
        clean=torch.load(cache/f"{row['base_id']}_clean.pt",weights_only=True)['input']
        maps=item['maps'].clone();residual=maps[0][None,None]
        maps[0]=(residual/(F.avg_pool2d(residual,15,stride=1,padding=7)+.03))[0,0]
        maps[:,:4]=0;maps[:,-4:]=0;maps[:,:,:4]=0;maps[:,:,-4:]=0
        combined=((maps-median)/scale).mean(0)
        panels=[('Normal source',clean,False),('Corrupted input',item['input'],False),('Diffusion reconstruction',item['reconstruction'],False),('Seed disagreement',item['uncertainty'],True),
                ('Localized pixel residual',maps[0],True),('Texture deficiency',maps[1],True),('Combined attribution',combined,True),('Corruption mask',item['mask'],False)]
        pdf=target/f'{family}.pdf';c=canvas.Canvas(str(pdf),pagesize=(900,540))
        c.setFont('Helvetica-Bold',15);c.drawString(30,512,f'Development example | {family} | severity 0.30')
        c.setFont('Helvetica',10);c.drawString(30,492,f"Score {float(row['score']):.3f}; clean-calibrated threshold {float(row['threshold']):.3f}; flag {float(row['score'])>float(row['threshold'])}")
        for i,(name,value,heat) in enumerate(panels):
            x=30+(i%4)*218;y=275 if i<4 else 56
            value=value.squeeze().float()
            maximum=1.
            if heat:
                value=value.clamp_min(0);maximum=max(1e-6,float(torch.quantile(value,.99)));value=value/maximum
            image=pil_image(value)
            if heat:image=ImageOps.colorize(image,'#102a43','#ffba49')
            c.setFont('Helvetica-Bold',10);c.drawString(x,y+187,name)
            c.drawImage(ImageReader(image),x,y,width=176,height=176)
            c.setFont('Helvetica',8)
            if heat:c.drawString(x,y-13,f'Display range: 0 to {maximum:.3g} (q99)')
        c.setFont('Helvetica',9);c.drawString(30,20,'Each map uses its own display scale. Attribution is not a segmentation guarantee; final holdout not used.')
        c.save()
        with pymupdf.open(pdf) as document:document[0].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(target/f'{family}.png')
        figures.append({'family':family,'case_id':row['case_id'],'figure':str((target/f'{family}.png').relative_to(ROOT)),'score':float(row['score']),'threshold':float(row['threshold'])})
    (source/'figures.json').write_text(json.dumps(figures,indent=2)+'\n')
    print(f'Saved {len(figures)} measured development figures')


if __name__=='__main__':main()
