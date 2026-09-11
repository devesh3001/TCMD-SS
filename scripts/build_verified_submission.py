"""Build a report and execute an evidence notebook without numerical model imports.

No model results are synthesized: all values come from saved experiments. The
notebook reruns cached-score analysis, not blocked training or DINO inference.
"""
import csv
import json
import os
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_cached_fusion import read, curves


def build_figures():
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    import pymupdf
    directory=ROOT/'results/figures'
    directory.mkdir(parents=True,exist_ok=True)
    predictions=read(ROOT/'outputs/evaluation/predictions.csv')
    curve=curves([int(float(r['is_anomaly'])) for r in predictions],[float(r['score']) for r in predictions])
    def line_plot(name,title,series,xlabel,ylabel,xmax=1.,ymax=1.):
        path=directory/f'{name}.pdf'
        c=canvas.Canvas(str(path),pagesize=(650,370))
        c.setTitle(title); c.setFont('Helvetica-Bold',14);c.drawString(65,340,title)
        c.setStrokeColor(colors.HexColor('#ced4dc'))
        c.setFont('Helvetica',9)
        for i in range(6):
            x=65+530*i/5;y=65+235*i/5
            c.line(65,y,595,y);c.setFillColor(colors.black)
            c.drawRightString(57,y-3,f'{ymax*i/5:.2f}')
            c.drawCentredString(x,48,f'{xmax*i/5:.2f}')
        palette=['#167d9a','#c85b36','#6264a7']
        for index,(label,points) in enumerate(series):
            c.setStrokeColor(colors.HexColor(palette[index%3]));c.setLineWidth(1.8)
            p=c.beginPath()
            for j,(x,y) in enumerate(points):
                (p.moveTo if j==0 else p.lineTo)(65+530*x/xmax,65+235*y/ymax)
            c.drawPath(p);c.setFillColor(colors.HexColor(palette[index%3]));c.drawString(70+index*175,315,label)
        c.setFillColor(colors.black);c.drawCentredString(330,22,xlabel)
        c.saveState();c.translate(15,185);c.rotate(90);c.drawCentredString(0,0,ylabel);c.restoreState()
        c.save()
        with pymupdf.open(path) as document:
            document[0].get_pixmap(matrix=pymupdf.Matrix(2,2)).save(directory/f'{name}.png')
    line_plot('baseline_roc','Figure 1. Historical development ROC', [('TCMD-SS baseline',[(p['fpr'],p['tpr']) for p in curve])],'False-positive rate','True-positive rate')
    line_plot('baseline_pr','Figure 2. Historical development precision-recall', [('TCMD-SS baseline',[(p['recall'],p['precision']) for p in curve])],'Recall','Precision')
    history=read(ROOT/'outputs/diffusion/training_history.csv')
    line_plot('training','Figure 3. Normal-only training and validation',[(k,[(float(r['epoch']),float(r[k])) for r in history]) for k in ['loss','normal_validation_l1']],'Epoch','Loss (different objectives)',8,.25)
    ablations=read(ROOT/'outputs/ablation_results.csv')
    line_plot('evolution','Figure 4. Shared-checkpoint scoring ablations',[('AUROC',[(i+1,float(r['auroc'])) for i,r in enumerate(ablations[:3])])],'Version: 1 pixel, 2 pixel + DINO, 3 four branches','Development AUROC',3,1.)


def build_pdf():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4,landscape
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
    directory=ROOT/'report';directory.mkdir(exist_ok=True)
    target=directory/'TCMD_SS_Report.pdf'
    if target.exists():
        if '--revise-draft' not in sys.argv:
            raise FileExistsError('Preserve the existing report; use --revise-draft to archive this draft first')
        import hashlib
        archive=ROOT/'tmp/pdfs/draft_archive'
        archive.mkdir(parents=True,exist_ok=True)
        (archive/f'report-{hashlib.sha256(target.read_bytes()).hexdigest()[:12]}.pdf').write_bytes(target.read_bytes())
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body',fontName='Helvetica',fontSize=11,leading=16,spaceAfter=10))
    styles.add(ParagraphStyle(name='CaptionSmall',fontSize=9,leading=12,spaceAfter=8))
    story=[]
    markdown=[]
    def text(value,style='Body'):
        story.append(Paragraph(value,styles[style]))
        import re
        markdown.append(('## ' if style=='Heading1' else '')+re.sub('<[^>]+>','',value))
    def table(headers,rows,widths):
        markdown.append('| '+' | '.join(map(str,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows))
        values=[[Paragraph(escape(str(x)),styles['CaptionSmall']) for x in row] for row in [headers,*rows]]
        t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#748da0')),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,12)])
    def figure(path,width=710):
        markdown.append(f'![Development figure](../{path.relative_to(ROOT).as_posix()})')
        from PIL import Image as PILImage
        with PILImage.open(path) as image:
            height=width*image.height/image.width
        story.append(Image(str(path),width=width,height=height))
    def page(title):
        if story:story.append(PageBreak())
        text(title,'Heading1')
    metrics=json.loads((ROOT/'outputs/evaluation/metrics.json').read_text())
    ablations=read(ROOT/'outputs/ablation_results.csv')
    split=json.loads((ROOT/'results/development/split_protocol.json').read_text())
    cached=json.loads((ROOT/'results/development/cached_fusion_v5/calibration_and_selection.json').read_text())
    page('TCMD-SS | HiRISE anomaly detection')
    text('Unsupervised deep generative modeling on Martian surface imagery','Heading2')
    text('<b>Evidence-backed development report. Final holdout NOT evaluated.</b>')
    text('I study structural faults in normal Mars imagery using masked conditional diffusion. I compare reconstructed terrain with the observed image and evaluate the resulting anomaly scores and maps. The historical baseline includes independent auxiliary branches. The new three-channel generative discrepancy has now been evaluated on a separate six-family development protocol; its DINO semantic extension remains untested.')
    table(['Measured system','AUROC','TPR','Clean FPR'],[['Historical fixed fusion','0.8893','0.6581','5.47%'],['Historical equal-weight ablation','0.9011','0.6458','3.91%'],['New cached DINO + spectral (diagnostic only)','0.9327','0.9081','25.00%']],[350,100,100,160])
    text('The last row excludes diffusion and has unacceptable false positives for the intended use. It is not selected as the competition model. The rows use different normal calibration protocols and must not be presented as controlled proof of one model improving another.')
    text('Research scope: the supplied MarsPS practice statement. I treat all eight terrain classes as normal and reserve final evaluation until development is complete. No GitHub publication or official competition submission has occurred.')
    page('1 | Dataset integrity and experimental separation')
    text('Official NASA/JPL HiRISE v3.2: 64,947 grayscale JPEG images, 227 x 227 pixels, 10,815 original landmarks and 232 source observations. Archive MD5 verified: 236d9c627db1a5970e77a01a8c8a035a. Rotations, flips and brightness variants remain grouped with their original.')
    table(['Role','Images','Unique bases','Sources'],[[n,v['images'],v['base_id'],v['source_id']] for n,v in split['counts'].items()],[300,130,140,140])
    text('No source, base, exact/near-linked group crosses the revised roles. Previously evaluated sources are development. The 71 final originals from three sources remain reserved; their eligibility is based on saved workspace exposure history, not unknown external experiments.')
    text('Audit: no exact byte/pixel duplicate groups or corrupt images. Perceptual hashing found 142 cross-base candidate pairs; these include possible false positives. Conservative linked grouping is retained. Only three final sources means high uncertainty and restricted terrain coverage.')
    page('2 | All semantic terrain classes are normal')
    classes=read(ROOT/'outputs/class_distribution.csv')
    table(['Class ID','Class name','All images','Original images','Augmented images'],[[r[k] for k in list(classes[0])[:5]] for r in classes],[100,190,140,140,140])
    text('Terrain labels describe craters, dunes and other legitimate Mars structures. They are not anomaly labels. Training and calibration use clean records only. Synthetic faults are confined to development/stress evaluation.')
    text('The historical benchmark has 128 clean images and 1,056 corrupted cases derived from 32 originals. Its 11 corruption kinds use severities 0.2, 0.5 and 0.8. Blur-plus-noise is not a pure local-blur test, and checkerboard contamination is not a representative sample of real foreign imagery.')
    page('3 | Generative model and scoring methodology')
    text('<b>Trained baseline:</b> custom 570,497-parameter grayscale U-Net at 128 x 128. Inputs are noisy hidden pixels, visible context and a mask. Masked noise-prediction MSE uses only normal Mars. Four complementary inference masks reconstruct every pixel while hiding the query region from conditioning.')
    text('Eight epochs completed, 1,024 optimizer steps and 4,096 sample presentations. Training pool: 921 originals; normal validation pool: 103, with eight scored for checkpoint selection. Best normal-validation reconstruction MAE: 0.10808. These measurements describe the saved bounded local experiment.')
    text('Baseline semantic scores use frozen satellite DINOv3 patch memory; local FFT and global latent distance are separate auxiliary branches. Fixed fusion is 0.40 / 0.25 / 0.20 / 0.15 after robust calibration, with a historical clean 99th-percentile threshold. These independent branches are not the new generative-discrepancy architecture.')
    text('<b>Generative revision:</b> observed image -> two-seed masked diffusion -> mean reconstruction and seed disagreement -> pixel, texture and directional spectral discrepancies -> clean-A robust z -> clean-B q95 threshold. Every tested discrepancy references the generator output. The proposed corresponding-patch DINOv2 extension remains blocked and is not included in the reported three-channel scores.')
    text('Single clean training-image reconstruction smoke: 15 DDIM steps, seeds 42/1042, 3.35 seconds, residual MAE 0.06196. Seed disagreement is not a calibrated uncertainty interval. The frozen final-run guard is implemented but has not been activated.')
    page('4 | Normal-only learning curves')
    figure(ROOT/'results/figures/training.png',700)
    text('Training noise MSE and validation reconstruction MAE have different meanings. The minimum validation reconstruction error occurs at epoch 6; later validation degradation supports preserving the best checkpoint. The scored validation subset is small.','CaptionSmall')
    page('5 | Historical baseline results')
    table(['Metric','Value'],[[k,f'{metrics[k]:.4f}'] for k in ['auroc','auprc','f1','precision','recall','false_positive_rate']],[350,360])
    text('AUROC 95% interval: 0.8609-0.9170 from 300 bootstrap resamples of base landmarks. This interval does not account for source-level dependence or architecture selection. AUPRC and precision reflect the artificially high anomaly prevalence (1,056/1,184), not deployment prevalence.')
    text('Historical decision threshold: 1.7794. Confusion matrix: TN 121, FP 7, FN 361, TP 695. Saved evaluation time: 1,369.5 seconds / 1,184 cases = 1.16 seconds per case, including scoring overhead. This is not a controlled production-latency benchmark.')
    page('6 | ROC: threshold-independent ranking')
    figure(ROOT/'results/figures/baseline_roc.png',700)
    page('6 | Precision-recall: benchmark prevalence matters')
    figure(ROOT/'results/figures/baseline_pr.png',700)
    page('7 | Family-level strengths and failures')
    family=read(ROOT/'outputs/evaluation/by_family.csv')
    table(['Family','AUROC','TPR'],[[r['family'],f"{float(r['auroc']):.4f}",f"{float(r['recall']):.4f}"] for r in family],[430,140,140])
    text('Patch duplication is weakest (AUROC 0.6637, TPR 0.1563). Stripes and synthetic contamination are strong under this benchmark. Pure blur, subtle sensor faults and realistic contamination still need the requested six-family development protocol.')
    page('8 | Model evolution and retained negative evidence')
    table(['Shared-checkpoint variant','AUROC','TPR','FPR'],[[r['variant'],f"{float(r['auroc']):.4f}",f"{float(r['recall']):.4f}",f"{float(r['false_positive_rate']):.4f}"] for r in ablations],[350,120,120,120])
    text('Versions 1-3 are scoring ablations on one diffusion checkpoint, not three independently trained models. Removing latent scoring slightly improves AUROC; equal weighting is the best historical diffusion-containing ablation. Both were inspected, so they remain development evidence. MODEL_EVOLUTION.md records symptoms, diagnoses, fixes and measurements.')
    page('9 | New source-independent calibration experiment')
    text('Cached scores were normalized using 71 clean calibration-A images. Each mean/max expert combination was thresholded at the 95th percentile of 57 source-disjoint clean calibration-B images. No anomaly labels set a threshold. Thirteen discrete ablations were compared on already-consumed development images.')
    revised=read(ROOT/'results/development/cached_fusion_v5/ablations.csv')
    chosen=['pixel_only_mean','DINO_only_mean','spectral_only_mean','DINO_spectral_mean','pixel_DINO_mean','pixel_spectral_mean','pixel_DINO_spectral_mean','equal_four_mean']
    table(['Cached-score variant','AUROC','TPR','FPR'],[[r['variant'],f"{float(r['auroc']):.4f}",f"{float(r['tpr']):.4f}",f"{float(r['fpr']):.4f}"] for r in revised if r['variant'] in chosen],[350,120,120,120])
    text('<b>Decision: do not deploy the highest-AUROC diagnostic.</b> Its 25% clean FPR and absence of diffusion make it unsuitable. Pixel + DINO + spectral reaches 0.8981 AUROC but 17.97% FPR. Larger source-diverse clean calibration and actual discrepancy development are needed; q95 on a small calibration subset does not guarantee 5% FPR on new sources.')
    page('10 | Explanation example: flagged missing crop')
    figure(ROOT/'outputs/heatmaps/example_05.png',710)
    text('Historical saved diagnostic panels. The displayed image/map/branch scores are real saved outputs. Heatmaps are attribution aids, not ground-truth segmentations. The latent branch is global and cannot supply localized evidence.','CaptionSmall')
    page('11 | Failure review: high-scoring clean terrain')
    figure(ROOT/'outputs/heatmaps/example_08.png',710)
    text('A high-scoring normal example illustrates why natural terrain diversity must not be relabelled as anomaly. Qualitative maps alone cannot establish localization accuracy. Pixel AUROC/IoU for the revised method have not been measured.','CaptionSmall')
    if (ROOT/'results/development/generative_v7/summary.json').exists():
        v6=json.loads((ROOT/'results/development/generative_v6/summary.json').read_text())
        v7=json.loads((ROOT/'results/development/generative_v7/summary.json').read_text())
        page('12 | Six-family generative development experiment')
        text('I evaluated 16 previously consumed normal originals and 288 deterministic corruptions: stripes, dead pixels, missing patches, pure local blur, patch duplication and procedural foreign content, each at 0.15, 0.30 and 0.50 severity. Separate sets of 64 clean images fit normalization and q95 thresholds. No anomaly was used to train the generator or set a threshold.')
        a=v6['complete_three_max'];b=v7['best_diagnostic']
        table(['Development revision','AUROC','TPR','FPR','F1'],[['V6: initial three-channel maximum',*[f'{a[k]:.4f}' for k in ['auroc','tpr','fpr','f1']]],['V7: refined three-channel mean',*[f'{b[k]:.4f}' for k in ['auroc','tpr','fpr','f1']]]],[310,100,100,100,100])
        text('V7 normalizes residuals against local reconstruction mismatch and compares narrow spectral peaks against neighboring frequency energy. I also replace degenerate zero-MAD map scaling with a clean-data spread estimate. Mean and maximum fusion are compared on development; the final holdout is not involved.')
        text(f"Bounded inference: {v6['mean_inference_seconds']:.3f} seconds per image averaged across 432 scored images, including two seeds and three discrepancy channels; peak allocated GPU memory {v6['peak_gpu_mib']:.1f} MiB. GPU: RTX 3050 Laptop. Cached refinement requires no new reconstruction.")
        text('<b>Interpretation:</b> AUROC improves within this six-family study, but remains below 0.80 and clean FPR is 2/16. The 0.7181 result cannot be compared directly with historical 0.9011: the corruption types, strengths, calibration and sample sizes differ.')
        page('13 | Six-family strengths and remaining failures')
        family7=[r for r in read(ROOT/'results/development/generative_v7/by_family_severity.csv') if r['variant']=='complete_three_mean' and r['dimension']=='family']
        table(['Family','AUROC','TPR','Paired anomaly > clean'],[[r['value'],f"{float(r['auroc']):.4f}",f"{float(r['tpr']):.4f}",f"{float(r['paired_anomaly_gt_clean']):.4f}"] for r in family7],[300,110,110,190])
        loc=json.loads((ROOT/'results/development/generative_v7/localization_summary.json').read_text())
        interval=loc['source_bootstrap']
        text(f"Source-cluster bootstrap: {interval['sources']} development sources, {interval['replicates']} replicates, AUROC interval {interval['auroc_lower']:.4f}-{interval['auroc_upper']:.4f}. This interval is conditional on the development-selected method and the small available sample, not an unbiased final-performance interval.")
        text('The stored family/severity CSVs retain every attempted variant. Weak copy-move and foreign-content results remain visible; no final model is declared on the strength of a selected AUROC alone.')
        page('14 | Localization and normal-only map calibration')
        locrows=read(ROOT/'results/development/generative_v7/localization_by_family.csv')
        def number(x):return f'{float(x):.4f}' if x else 'Not defined'
        table(['Family','Pixel AUROC','IoU at clean pixel q95'],[[r['family'],number(r['pixel_auroc']),number(r['iou_at_clean_pixel_q95'])] for r in locrows],[350,180,180])
        text('The image threshold and pixel-map threshold are distinct, each calibrated using clean data. Full-image stripe masks have no background, so pixel AUROC is undefined. Spectral profile maps are directional attribution, not inverse-FFT localization. Masks describe procedural corruptions, not annotated real sensor failures.')
        text('Dead-pixel localization improved after replacing a near-zero texture scale that previously dominated the map. Copy-move remains near chance, and blur/foreign-content localization is insufficient for strong segmentation claims.')
        page('15 | New generative explanation: dead pixels')
        figure(ROOT/'results/figures/generative_v7/dead_pixels.png',710)
        page('16 | New generative failure review: copied terrain')
        figure(ROOT/'results/figures/generative_v7/patch_duplication.png',710)
        page('17 | Rare-anomaly sensitivity and related work')
        rare=[r for r in read(ROOT/'results/development/rare_prevalence/sensitivity.csv') if float(r['assumed_anomaly_prevalence'])==.05]
        table(['Historical method','Expected precision at 5% anomalies','False alerts per 1,000'],[[r['variant'],f"{float(r['expected_precision']):.4f}",f"{float(r['expected_false_alerts_per_1000']):.1f}"] for r in rare],[300,220,190])
        text('VAD4Space motivates evaluating rare anomalies and computational cost. I use analytical prevalence reweighting of existing development scores; I do not import its rover/lunar datasets or geological anomaly labels. No threshold changes here. This calculation assumes unchanged class-conditional score distributions, so it is not a real deployment benchmark.')
        text('Related work: Genilotti et al., VAD4Space: Visual Anomaly Detection for Planetary Surface Imagery, 2026, https://arxiv.org/abs/2603.13993. Its feature-based benchmarks are complementary references; the central model here remains a normal-only generator.','CaptionSmall')
    page('18 | Requirements, limitations and reproducibility')
    table(['Practice-statement requirement','Evidence / status'],[['Normal-only deep generative model','Trained diffusion checkpoint and normal manifests preserved.'],['Flag and explain anomalies','Historical scoring, threshold and saved heatmaps; synthetic-only validation.'],['Documented notebook with code and outputs','notebooks/TCMD_SS_Final.ipynb executes artifact checks and cached analysis; model rerun is blocked.'],['PDF with labelled results','This development report; no untouched final claim.'],['At least three model versions','MODEL_EVOLUTION.md: actual shared-checkpoint ablations and candidate revisions.'],['GitHub repository link','Local files prepared; destination and publication not configured.']],[270,440])
    text('Runtime: Windows Application Control blocks numpy.random._pcg64 and DINO import fails. Fourteen isolated tests pass; full-suite collection remains blocked. The Torch-only generative experiment ran on the GPU, but the DINO extension and complete final protocol remain unexecuted. No security policy was disabled.')
    text('Commands: python scripts/evaluate_cached_fusion.py; python -m unittest tests.test_revision tests.test_discrepancy tests.test_cached_fusion -v; python scripts/build_verified_submission.py. The notebook replays evidence checks and cached analysis. Preserve final-holdout manifests; complete development and freeze configuration, model and calibration hashes before the one permitted final evaluation.')
    text('Data source: NASA/JPL HiRISE v3.2, Zenodo record 4002935 (https://zenodo.org/records/4002935). Requirements source: user-provided marsps.pdf, marked as a practice statement. Saved tables/checkpoints, not the illustrative numbers in prior notes, support every reported measurement.','CaptionSmall')
    def footer(c,doc):
        c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#546477'))
        c.drawString(42,22,'TCMD-SS | Development evidence | Final holdout not evaluated')
        c.drawRightString(800,22,str(doc.page))
    SimpleDocTemplate(str(target),pagesize=landscape(A4),leftMargin=48,rightMargin=48,topMargin=34,bottomMargin=36).build(story,onFirstPage=footer,onLaterPages=footer)
    (directory/'TCMD_SS_Report.md').write_text('\n\n'.join(markdown)+'\n',encoding='utf-8')
    return target


def build_evidence_notebook():
    import nbformat as nbf
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    path=ROOT/'notebooks/TCMD_SS_Final.ipynb'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        if '--revise-draft' not in sys.argv:
            raise FileExistsError('Preserve existing notebook')
        import hashlib
        archive=ROOT/'tmp/pdfs/draft_archive'
        archive.mkdir(parents=True,exist_ok=True)
        (archive/f'notebook-{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}.ipynb').write_bytes(path.read_bytes())
    cells=[]
    md=lambda value:cells.append(nbf.v4.new_markdown_cell(value))
    code=lambda value:cells.append(nbf.v4.new_code_cell(value))
    md('# TCMD-SS: HiRISE anomaly detection\n\n**Development evidence; final holdout not evaluated.** I treat all eight Mars terrain classes as normal and investigate structural image faults using masked diffusion. This notebook checks saved experiments and reproduces score analysis. The three-channel generative experiment ran on the GPU; the DINO extension remains blocked.')
    code("from pathlib import Path\nimport csv, json, sys\nfrom IPython.display import display, Image, Markdown\nROOT=Path.cwd()\nassert (ROOT/'outputs/dataset_audit.json').exists()\nsys.path.insert(0,str(ROOT))\nfrom scripts.evaluate_cached_fusion import read, metrics, main as evaluate_cached\nfrom scripts.prepare_revision import disjoint\nprint('Repository:', ROOT)")
    md('## Dataset and labels\nThe archive MD5 was verified during acquisition. Augmentations, original landmarks, source observations and duplicate-linked components are grouped. Semantic labels are not anomaly labels.')
    code("audit=json.loads((ROOT/'outputs/dataset_audit.json').read_text())\nprint({k:audit[k] for k in ['image_count','source_observation_count','base_landmark_count','dimensions']})\ndisplay(read(ROOT/'outputs/class_distribution.csv'))")
    md('## Exposure-aware separation\nEvery previously evaluated source is development. No final-holdout image is loaded by this notebook. Unknown external exposure remains a limitation.')
    code("names=['train_normal','calibration_a','calibration_b','development','final_holdout']\nparts={n:read(ROOT/f'data/splits/generative/{n}.csv') for n in names}\ndisjoint(parts)\nprint('Source/base/linked-group overlap: zero')\ndisplay(json.loads((ROOT/'results/development/split_protocol.json').read_text())['counts'])")
    md('## Model and training\nThe baseline is a 570,497-parameter custom masked U-Net trained on normal Mars only. Frozen satellite DINOv3, local FFT and latent distance are historical auxiliary scores. The new three-channel development study compares observed versus generated pixel, texture and directional spectra; semantic discrepancy remains pending.')
    code("display(json.loads((ROOT/'outputs/diffusion/training_summary.json').read_text()))\ndisplay(read(ROOT/'outputs/diffusion/training_history.csv'))\ndisplay(Image(filename=str(ROOT/'results/figures/training.png')))")
    md('## Training and inference code\nThe saved modules contain implementation, not pseudocode. Importing the DINO/training pipeline is currently blocked by the documented runtime failure. The following cell displays the relevant source; it does not execute training.')
    code("display(Markdown('```python\\n'+(ROOT/'src/generative/discrepancy.py').read_text(encoding='utf-8')+'\\n```'))\nprint('Baseline training command: python scripts/train_diffusion.py --config configs/local.yaml')\nprint('Do not rerun into existing experiment outputs; use a new output directory.')")
    md('## Reproduce saved baseline metrics\nRecompute ranking and threshold metrics from saved predictions with a tied-score-safe implementation. Baseline AP reflects artificial anomaly prevalence. Existing scores were inspected and are development evidence.')
    code("rows=read(ROOT/'outputs/evaluation/predictions.csv')\nreference=json.loads((ROOT/'outputs/evaluation/metrics.json').read_text())\nrecomputed=metrics(rows,[float(r['score']) for r in rows],reference['threshold'])\nfor k in ['auroc','auprc','f1','precision','recall']:\n    assert abs(reference[k]-recomputed[k])<1e-12\ndisplay(recomputed)\ndisplay(Image(filename=str(ROOT/'results/figures/baseline_roc.png')))\ndisplay(Image(filename=str(ROOT/'results/figures/baseline_pr.png')))")
    md('## New cached development experiment\nFit robust normalization on clean calibration A; set each q95 threshold on independent clean calibration B. Evaluate 13 discrete scoring ablations on already-consumed development only. This is a calibration/fusion experiment, not new generative-discrepancy inference.')
    code("evaluate_cached()\ndisplay(read(ROOT/'results/development/cached_fusion_v5/ablations.csv'))")
    md('## Selection decision\nThe highest AUROC (0.9327) comes from DINO + spectral without diffusion and has 25% clean FPR. It is not adopted. Diffusion + DINO + spectral has 0.8981 AUROC and 17.97% FPR. Source-diverse clean calibration and the requested six-family experiment remain necessary. Historical equal weighting has 0.9011 AUROC and 3.91% FPR under a different calibration protocol.')
    code("display(read(ROOT/'outputs/evaluation/by_family.csv'))\ndisplay(read(ROOT/'outputs/evaluation/by_severity.csv'))\ndisplay(read(ROOT/'outputs/ablation_results.csv'))")
    md('## Explanation and failure examples\nThese panels are saved historical explanations. They are not new semantic-discrepancy heatmaps or validated segmentation results.')
    code("examples=json.loads((ROOT/'outputs/heatmaps/explanations.json').read_text())\nfor e in [examples[5],examples[-1]]:\n    display(Markdown(e['explanation']))\n    display(Image(filename=str(ROOT/e['figure'])))")
    md('## Measured model evolution\nVersions 1-3 are same-checkpoint scoring ablations. Candidate reconstruction and subsequent cached calibration are explicitly distinguished from trained systems.')
    code("display(Markdown((ROOT/'MODEL_EVOLUTION.md').read_text(encoding='utf-8')))")
    if (ROOT/'results/development/generative_v7/summary.json').exists():
        md('## Six-family generative experiment: V6 to V7\nI keep calibration A/B source-disjoint and use the same 16 consumed development originals for comparison. I retain negative results. These scores are not comparable to the earlier, stronger-corruption benchmark.')
        code("v6=json.loads((ROOT/'results/development/generative_v6/summary.json').read_text())\nv7=json.loads((ROOT/'results/development/generative_v7/summary.json').read_text())\ndisplay(v6)\ndisplay(v7)\nnewrows=[r for r in read(ROOT/'results/development/generative_v7/predictions.csv') if r['variant']=='complete_three_mean']\ncheck_metrics=metrics(newrows,[float(r['score']) for r in newrows],float(newrows[0]['threshold']))\nassert abs(check_metrics['auroc']-v7['best_diagnostic']['auroc'])<1e-12\ndisplay(check_metrics)\nprint('Reproduce GPU experiment: python scripts/run_generative_development.py')\nprint('Reproduce cached refinement: python scripts/refine_generative_scores.py')")
        md('## Localization and failure analysis\nClean A determines map normalization; independent clean B supplies the pixel q95 threshold. Full-image stripe masks have no background for pixel AUROC. Maps are explanatory diagnostics rather than validated real-fault segmentations.')
        code("display(read(ROOT/'results/development/generative_v7/localization_by_family.csv'))\ndisplay(json.loads((ROOT/'results/development/generative_v7/localization_summary.json').read_text()))\nfor family in ['dead_pixels','patch_duplication']:\n    display(Image(filename=str(ROOT/f'results/figures/generative_v7/{family}.png')))")
        md('## Rare-anomaly sensitivity\nInspired by VAD4Space (Genilotti et al., 2026, https://arxiv.org/abs/2603.13993), I examine how precision changes with assumed anomaly prevalence. This is analytical reweighting of existing scores, not another dataset or new deployment evidence.')
        code("from scripts.evaluate_rare_prevalence import main as rare_analysis\nrare_analysis()")
    md('## Validation and final status\nThe full numerical suite is blocked at collection. Isolated protocol, discrepancy and cached-metric tests execute below. No final-run claim is created, and architecture is not frozen. Complete the six-family development benchmark and freeze all selected artifacts before evaluating the holdout once.')
    code("import subprocess\ncheck=subprocess.run([sys.executable,'-m','unittest','tests.test_revision','tests.test_discrepancy','tests.test_cached_fusion','tests.test_stress','tests.test_generative_diagnostics','-v'],cwd=ROOT,capture_output=True,text=True)\nprint(check.stdout+check.stderr)\nassert check.returncode==0\nassert not (ROOT/'results/final/final_run_claim.json').exists()\nprint('Final evaluation: not run. GitHub publication: not performed.')")
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'TCMD-SS Python','language':'python','name':'tcmdss'}})
    kernel_dir=ROOT/'outputs/jupyter/kernels/tcmdss';kernel_dir.mkdir(parents=True,exist_ok=True)
    (kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'TCMD-SS Python','language':'python'}))
    runtime=ROOT/'outputs/jupyter/runtime';runtime.mkdir(parents=True,exist_ok=True)
    os.environ['JUPYTER_RUNTIME_DIR']=str(runtime);os.environ['IPYTHONDIR']=str(ROOT/'outputs/jupyter/ipython')
    manager=KernelManager(kernel_name='tcmdss',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
    client=NotebookClient(nb,timeout=180,km=manager,resources={'metadata':{'path':str(ROOT)}})
    client.execute();nbf.write(nb,path)
    primary=ROOT/'TCMD_SS_HiRISE_Anomaly_Detection.ipynb'
    if not primary.exists():nbf.write(nb,primary)
    elif '--revise-draft' in sys.argv:
        (archive/'root_notebook_previous.ipynb').write_bytes(primary.read_bytes())
        nbf.write(nb,primary)
    return path


def build_notebook():
    # Keep report rebuilds from replacing the expanded walkthrough with the old summary.
    from scripts.build_walkthrough_notebook import build
    return build()


if __name__=='__main__':
    if '--notebook-only' not in sys.argv:
        build_figures();print(build_pdf())
    print(build_notebook())
