"""Create a readable, inline Kaggle/Colab-style research notebook and execute review mode."""
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
try:
    from scripts.structure_notebook import structure_notebook
except ModuleNotFoundError:
    from structure_notebook import structure_notebook

ROOT=Path(__file__).resolve().parents[1]


def build():
    cells=[]
    def md(value):cells.append(nbf.v4.new_markdown_cell(value.strip()))
    def code(value,scope='review'):
        cells.append(nbf.v4.new_code_cell(value.strip(),metadata={'tags':[scope]}))
    def definitions(path,*names):
        source=(ROOT/path).read_text(encoding='utf-8');lines=source.splitlines()
        tree=ast.parse(source)
        for name in names:
            node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef)) and n.name==name)
            start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
            snippet='\n'.join(lines[start:node.end_lineno])
            if name=='frequency_peaks':snippet=snippet.replace('torch.hann_window(128)','torch.hann_window(128, device=images.device)')
            if name=='stress':
                snippet=snippet.replace('list(image.get_flattened_data())','list(image_pixels(image))')
                snippet=snippet.replace('list(image.filter(ImageFilter.GaussianBlur(.5+5*severity)).get_flattened_data())','list(image_pixels(image.filter(ImageFilter.GaussianBlur(.5+5*severity))))')
            code(snippet,'implementation')
    md('''# TCMD-SS: finding structural faults in HiRISE images

### Data, training and evaluation

All Mars terrain classes are **normal**. The task is to detect image faults: dead pixels, stripes, missing regions and copied content.

Masked conditional diffusion reconstructs hidden terrain. Differences between the input and reconstruction provide the anomaly score.

**Current evidence:** the refined three-channel model reached **0.7181 development AUROC** on the six-family study. The historical four-branch benchmark reached 0.9011 for equal weighting, but used different corruptions and is not directly comparable. **The final holdout has not been evaluated.**

Saved results are labelled. Training is optional.''')
    md('''## Notebook roadmap

1. Environment and experiment controls
2. Dataset inspection, class balance and augmentation families
3. Source-disjoint manifests and PyTorch data loading
4. U-Net, diffusion loss and complementary reconstruction
5. Normal-only training and checkpoint review
6. Six development corruption families
7. Pixel, texture, spectral and optional DINO discrepancies
8. Clean calibration, development inference and metrics
9. Ablations, localization, rare-anomaly sensitivity and failures
10. Conclusions and final-holdout safeguards''')
    md('''## 1. Environment and controls

On **Colab**, enable a GPU runtime and upload/mount the project folder. On **Kaggle**, attach the project/evidence folder and the existing HiRISE dataset, then select a GPU accelerator. Set the two path overrides below when the folders are not automatically found.

The notebook needs the audited manifests and saved results to reproduce the existing study. It does not silently download another dataset or invent a GitHub clone URL. Review mode executes definitions, checks, examples and metric recomputation. Expensive fresh training/inference is opt-in and writes to a new run directory.''')
    code('''from pathlib import Path
import sys, os, json, csv, math, random, statistics, hashlib, html, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from IPython.display import display, HTML, SVG, Markdown, Image as DisplayImage

PROJECT_ROOT_OVERRIDE = None   # Example: Path('/content/TCMD-SS')
DATASET_ROOT_OVERRIDE = None   # Folder containing map-proj-v3_2/
INSTALL_DEPENDENCIES = False
RUN_TRAINING = False
RUN_DEVELOPMENT_INFERENCE = False
RUN_DINO = False
RUN_LIVE_RECONSTRUCTION = True
SEED = 42
print('Mode: review. Training, full development inference and DINO are opt-in.')''')
    code('''if INSTALL_DEPENDENCIES:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
        'torch', 'torchvision', 'numpy', 'pandas', 'Pillow', 'scikit-learn',
        'ImageHash', 'PyYAML', 'timm', 'safetensors', 'matplotlib', 'tqdm'])
else:
    print('Using the installed environment; no packages were changed.')''','opt-in')
    code('''candidates = [Path.cwd(), *Path.cwd().parents]
for base in [Path('/content'), Path('/kaggle/working'), Path('/kaggle/input')]:
    if base.exists():
        candidates.extend([base, *[p for p in base.iterdir() if p.is_dir()]])
matches = list(dict.fromkeys(p.resolve() for p in candidates
    if (p/'outputs/dataset_audit.json').exists()
    and (p/'data/splits/generative/train_normal.csv').exists()))
if PROJECT_ROOT_OVERRIDE is not None:
    ROOT = Path(PROJECT_ROOT_OVERRIDE).resolve()
elif len(matches) == 1:
    ROOT = matches[0]
else:
    raise RuntimeError(f'Set PROJECT_ROOT_OVERRIDE. Evidence-folder candidates: {matches}')
assert (ROOT/'outputs/dataset_audit.json').exists(), 'Attach the project evidence folder.'
print('Selected project:', ROOT)

raw_candidates = [ROOT/'data/raw/hirise_v3_2']
if Path('/kaggle/input').exists():
    raw_candidates.extend(p.parent for p in Path('/kaggle/input').glob('**/map-proj-v3_2') if p.is_dir())
raw_candidates = list(dict.fromkeys(p.resolve() for p in raw_candidates if (p/'map-proj-v3_2').is_dir()))
if DATASET_ROOT_OVERRIDE is not None:
    DATA_ROOT = Path(DATASET_ROOT_OVERRIDE).resolve()
elif len(raw_candidates) == 1:
    DATA_ROOT = raw_candidates[0]
else:
    raise RuntimeError(f'Set DATASET_ROOT_OVERRIDE. HiRISE candidates: {raw_candidates}')
print('Dataset candidates:', raw_candidates)
print('Selected dataset:', DATA_ROOT)''')
    code('''import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageDraw, ImageOps
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print({'python':sys.version.split()[0], 'torch':torch.__version__,
       'device':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU', 'seed':SEED})''')
    md('''**Runtime note.** On the original Windows machine, Application Control blocks a NumPy extension and the DINO import fails. The core notebook uses Python, Pillow and PyTorch so its generative review can still run. This does not remove or bypass that restriction. Fresh training and DINO must be tested in a working authorized numerical environment. Exact cross-device equality is not guaranteed.''')
    code('''SIZE = 128
CFG = dict(seed=SEED, size=SIZE, base_channels=32, batch_size=4, epochs=8,
           max_steps_per_epoch=128, learning_rate=3e-4, diffusion_steps=100,
           sampling_steps=15, reconstruction_seeds=(42,1042), top_fraction=.005,
           calibration_images=64, development_originals=16, severities=(.15,.30,.50))
RUN_DIR = (Path('/kaggle/working') if Path('/kaggle/working').exists() else ROOT/'outputs') / ('notebook_run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f'))
print(json.dumps(CFG,indent=2))
print('Fresh outputs, only if enabled:', RUN_DIR)''')
    md('''## 2. Small helpers for readable outputs

Tables and simple SVG charts below are generated from the actual values in the notebook. They avoid a plotting-library dependency in the restricted local runtime; the analysis does not depend on their display format.''')
    code('''def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def show_table(rows, columns=None, limit=20):
    rows=list(rows)
    if not rows:
        print('No rows.'); return
    columns=columns or list(rows[0])
    def value(x):
        return f'{x:.4f}' if isinstance(x,float) else str(x)
    header=''.join('<th style="padding:7px;text-align:left">'+html.escape(k)+'</th>' for k in columns)
    body=''.join('<tr>'+''.join('<td style="padding:7px;border-bottom:1px solid #ddd">'+html.escape(value(r.get(k,'')))+'</td>' for k in columns)+'</tr>' for r in rows[:limit])
    display(HTML('<table style="border-collapse:collapse"><thead style="background:#e9f1f7"><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table>'))
    if len(rows)>limit: print(f'Showing {limit} of {len(rows)} rows.')

def show_figure(relative_path):
    path=ROOT/relative_path
    if path.exists(): display(DisplayImage(filename=str(path)))
    else: print('Optional saved figure not attached:', relative_path)

def bars(labels, values, title, xlabel='Value'):
    labels=list(labels); values=list(map(float,values)); maximum=max(values,default=1) or 1
    height=65+32*len(labels)
    shapes=[f'<text x="12" y="24" font-size="17" font-weight="bold">{html.escape(title)}</text>']
    for i,(label,value) in enumerate(zip(labels,values)):
        y=43+i*32
        shapes += [f'<text x="10" y="{y+16}" font-size="12">{html.escape(str(label))}</text>',
                   f'<rect x="205" y="{y}" width="{440*value/maximum}" height="23" fill="#267b99"/>',
                   f'<text x="{213+440*value/maximum}" y="{y+16}" font-size="12">{value:.3f}</text>']
    shapes.append(f'<text x="205" y="{height-3}" font-size="11">{html.escape(xlabel)}; bars start at zero</text>')
    display(SVG(f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" style="background:white;font-family:sans-serif">'+''.join(shapes)+'</svg>'))''','implementation')
    md('''## 3. Inspect the HiRISE dataset

I use NASA/JPL HiRISE v3.2 from [Zenodo record 4002935](https://zenodo.org/records/4002935). The archive MD5 was checked at acquisition. The full audit is preserved rather than rerun every time the notebook opens.''')
    code('''audit=read_json(ROOT/'outputs/dataset_audit.json')
download=read_json(ROOT/'outputs/download_verification.json')
show_table([{'item':key,'value':audit.get(key)} for key in
    ['image_count','base_landmark_count','source_observation_count','dimensions','channels','formats']])
display(download)''')
    code('''print('Dataset root contents:')
for path in sorted(DATA_ROOT.iterdir()):
    print(('DIR ' if path.is_dir() else 'FILE'), path.name)
classes=read_csv(ROOT/'outputs/class_distribution.csv')
show_table(classes, ['class_id','class_name','image_count','original_image_count','augmented_image_count'])''')
    code('''bars([r['class_name'] for r in classes], [r['image_count'] for r in classes],
     'Semantic class distribution — every class is normal', 'Images')''')
    md('''The large “other” class makes random image sampling misleading. Augmentation-rich landmarks can also dominate a small training subset. I sample originals for bounded experiments and preserve whole source/duplicate-linked groups across roles.''')
    md('''## 4. Augmentations, originals and duplicate checks

The known suffixes are rotations `r90/r180/r270`, flips `fh/fv`, and brightness `brt`. The parser below illustrates the observed naming convention. The saved audit—not this illustration alone—also joins duplicate candidates and source observations.''')
    code('''TRANSFORMS={'r90','r180','r270','fh','fv','brt'}
def base_identifier(filename):
    pieces=Path(filename).stem.lower().split('-')
    transformations=[]
    while len(pieces)>1 and pieces[-1] in TRANSFORMS:
        transformations.insert(0,pieces.pop())
    return '-'.join(pieces), transformations

for name in ['ESP_012373_0930_RED-0050.jpg','ESP_012373_0930_RED-0050-r90.jpg',
             'ESP_012373_0930_RED-0050-brt.jpg']:
    print(name,'->',base_identifier(name))
assert base_identifier('ESP_012373_0930_RED-0050-r90.jpg')[0] == base_identifier('ESP_012373_0930_RED-0050.jpg')[0]''','implementation')
    code('''def group_count(value):
    return len(value) if isinstance(value,(list,dict)) else value
show_table([{'check':k,'result':group_count(audit.get(k))} for k in
    ['augmentation_groups','exact_duplicate_groups','pixel_duplicate_groups','possible_leakage','unusual_files']])
print('Near-duplicate candidate counts:', audit.get('near_duplicate_pair_counts'))
print('Perceptual similarity is a candidate relationship, not proof of identical content.')''')
    md('''## 5. Source-disjoint experiment roles

I keep train, calibration A, calibration B, development and final holdout separate by source ID, base ID and linked group. Calibration A learns score normalization; calibration B sets thresholds. Previously evaluated sources belong to development. Only manifest metadata is inspected for the final reserve; no final image is loaded.''')
    code('''GROUP_KEYS=('source_id','base_id','group_id','split_group_id')
def assert_group_separation(parts):
    for key in GROUP_KEYS:
        seen=set()
        for name,rows in parts.items():
            values={r[key] for r in rows}
            if not values or '' in values or seen & values:
                raise ValueError(f'Missing or overlapping {key} in {name}')
            seen.update(values)

names=['train_normal','calibration_a','calibration_b','development','final_holdout']
parts={name:read_csv(ROOT/f'data/splits/generative/{name}.csv') for name in names}
assert_group_separation(parts)
show_table([{'role':name,'images':len(rows),**{key:len({r[key] for r in rows}) for key in ['base_id','source_id']}} for name,rows in parts.items()])
print('All four grouping constraints pass.')''','implementation')
    code('''exposed=read_csv(ROOT/'outputs/evaluation/predictions.csv')
exposed_sources={r['source_id'] for r in exposed}
final_sources={r['source_id'] for r in parts['final_holdout']}
assert not exposed_sources & final_sources
print('Previously evaluated source overlap with final reserve:',len(exposed_sources & final_sources))
print('Final reserve:',len(parts['final_holdout']),'originals from',len(final_sources),'sources')
print('This exposure check covers saved project history, not unknown external experiments.')''')
    md('''## 6. Load grayscale images and inspect normal terrain

The loader resolves manifest paths against the selected dataset root, so original Windows paths do not need to exist in Colab/Kaggle. It checks the normal-only training contract. The montage uses training images only.''')
    code('''def resolve_image(row):
    relative=Path(row['path'])
    path=(DATA_ROOT/relative).resolve()
    if not path.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError('Manifest path escaped the dataset root')
    if not path.exists(): raise FileNotFoundError(path)
    return path

def assert_clean(rows):
    if not rows or any(float(r['is_anomaly'])!=0 or r['origin']!='hirise_clean'
                       or int(float(r['class_id'])) not in range(8) for r in rows):
        raise ValueError('Fitting/calibration requires clean HiRISE records')

def image_pixels(image):
    return image.get_flattened_data() if hasattr(image,'get_flattened_data') else image.getdata()

def tensor_image(image):
    values=image.convert('L')
    pixels=values.get_flattened_data() if hasattr(values,'get_flattened_data') else values.getdata()
    return torch.tensor(list(pixels),dtype=torch.float32).reshape(1,values.height,values.width)/255

def pil_image(values):
    values=values.detach().cpu().squeeze()
    return Image.frombytes('L',(values.shape[1],values.shape[0]),bytes(values.clamp(0,1).mul(255).round().to(torch.uint8).flatten().tolist()))

class MarsImages(Dataset):
    def __init__(self, rows, size=128):
        assert_clean(rows)
        self.rows,self.size=rows,size
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        with Image.open(resolve_image(self.rows[index])) as source:
            return tensor_image(source.convert('L').resize((self.size,self.size),Image.Resampling.BILINEAR))''','implementation')
    code('''training_rows=read_csv(ROOT/'outputs/diffusion/training_manifest.csv')
validation_rows=read_csv(ROOT/'outputs/diffusion/validation_manifest.csv')
assert_clean(training_rows); assert_clean(validation_rows)
assert_group_separation({'fit':training_rows,'validation':validation_rows})
allowed={r['split_group_id'] for r in parts['train_normal']}
assert all(r['split_group_id'] in allowed for r in training_rows+validation_rows)
train_data=MarsImages(training_rows)
validation_data=MarsImages(validation_rows)
print('Actual historical fit/validation pools:',len(train_data),len(validation_data))
example_batch=torch.stack([train_data[i] for i in range(4)])
print('Batch:',tuple(example_batch.shape),'range:',float(example_batch.min()),float(example_batch.max()))''')
    code('''montage=Image.new('RGB',(4*170,2*198),'white')
draw=ImageDraw.Draw(montage)
for class_id in range(8):
    row=next(r for r in training_rows if int(float(r['class_id']))==class_id)
    with Image.open(resolve_image(row)) as source:
        tile=source.convert('RGB').resize((160,160))
    x=(class_id%4)*170; y=(class_id//4)*198
    montage.paste(tile,(x,y))
    draw.text((x,y+166),row['class_name'],fill='black')
display(montage)''','live-data')
    code('''family_groups=defaultdict(list)
for row in parts['train_normal']: family_groups[row['base_id']].append(row)
example_family=next(rows for rows in family_groups.values() if len(rows)>=7)
show_table([{'filename':Path(r['path']).name,'base_id':r['base_id'],'source_id':r['source_id']} for r in example_family])
family_montage=Image.new('RGB',(7*120,145),'white'); draw=ImageDraw.Draw(family_montage)
for i,row in enumerate(sorted(example_family,key=lambda r:r['path'])[:7]):
    with Image.open(resolve_image(row)) as source: family_montage.paste(source.convert('RGB').resize((112,112)),(i*120,0))
    draw.text((i*120,116),str(base_identifier(row['path'])[1] or ['original']),fill='black')
display(family_montage)''','live-data')
    md('''## 7. The masked conditional U-Net

The network receives three channels: noisy hidden pixels, visible context and the binary mask. It predicts diffusion noise. Group normalization keeps small-batch training practical. These are the actual architecture definitions used by the saved checkpoint.''')
    definitions('src/diffusion/model.py','Residual','MaskedUNet')
    code('''model=MaskedUNet(CFG['base_channels']).to(DEVICE)
parameter_count=sum(p.numel() for p in model.parameters())
assert parameter_count==570497
print(model)
print(f'Trainable diffusion parameters: {parameter_count:,}')''')
    md(r'''## 8. Masks, noise schedule and training objective

Training hides roughly 10–40% of an image using rectangles or irregular masks. Inference uses four complementary masks: every pixel is reconstructed while hidden once. The masked noise-prediction objective is

$$L=\frac{\sum M\,\lVert\epsilon_\theta(x_t,x\odot(1-M),M,t)-\epsilon\rVert^2}{\sum M}.$$

The original hidden pixels never enter the visible-context channel.''')
    definitions('src/diffusion/process.py','training_masks','covering_masks','schedule','denoising_loss')
    code('''masks=covering_masks(SIZE)
assert torch.equal(masks.sum(0),torch.ones(1,SIZE,SIZE))
mask_montage=Image.new('L',(SIZE*4,SIZE))
for i,mask in enumerate(masks): mask_montage.paste(pil_image(mask),(SIZE*i,0))
display(mask_montage)
print('Four phases; hidden pixels per phase:',[int(m.sum()) for m in masks])
print('Every pixel hidden exactly once:',bool((masks.sum(0)==1).all()))''','live-check')
    code('''# Gradient sanity check on normal training images; no optimizer step is taken.
random.seed(SEED); torch.manual_seed(SEED)
model.train(); model.zero_grad(set_to_none=True)
loss=denoising_loss(model,example_batch[:2].to(DEVICE),CFG['diffusion_steps'])
loss.backward()
assert torch.isfinite(loss)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print('Untrained-model masked loss (sanity check, not a training result):',float(loss.detach()))
model.zero_grad(set_to_none=True)''','live-check')
    md('''## 9. DDIM reconstruction and seed disagreement

Each hidden tile starts from pure Gaussian noise. Visible context is retained throughout inference. I run two deterministic seeds and average their reconstructions. Their disagreement is a diagnostic of generation instability, not a calibrated confidence interval.''')
    definitions('src/diffusion/process.py','reconstruct')
    definitions('src/generative/discrepancy.py','normal_reconstruction')
    md('''## 10. Normal-only training loop

The loop below is an inline notebook wrapper around the same architecture and masked loss. It writes to a new run folder. The saved historical run used `scripts/train_diffusion.py`; its history is loaded in the next section. Defining this function does **not** mean a new training run occurred.

For a fresh run, set `RUN_TRAINING=True` near the top after checking the environment and paths. Checkpoints from this project are trusted local artifacts; do not load unknown pickle checkpoints.''')
    code('''def train_normal_model(fit_rows, val_rows, config, destination, resume=None):
    assert_clean(fit_rows); assert_clean(val_rows)
    assert_group_separation({'fit':fit_rows,'validation':val_rows})
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    random.seed(config['seed']); torch.manual_seed(config['seed'])
    network=MaskedUNet(config['base_channels']).to(DEVICE)
    optimizer=torch.optim.AdamW(network.parameters(),lr=config['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=DEVICE.type=='cuda')
    fit_loader=DataLoader(MarsImages(fit_rows),batch_size=config['batch_size'],shuffle=True,num_workers=0)
    val_loader=DataLoader(MarsImages(val_rows[:8]),batch_size=config['batch_size'],num_workers=0)
    history=[]; first_epoch=0; best=float('inf'); stale=0
    if resume is not None:
        state=torch.load(resume,map_location=DEVICE,weights_only=False)
        if state['training_bases'] != [r['base_id'] for r in fit_rows]:
            raise ValueError('Resume training subset changed')
        for key in ['base_channels','diffusion_steps','size']:
            if state['config'][key]!=config[key]: raise ValueError(f'Resume mismatch: {key}')
        network.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        history=state['history'];first_epoch=state['epoch']+1;best=state['best']
        torch.set_rng_state(state['torch_rng'].cpu());random.setstate(state['python_rng'])
        if DEVICE.type=='cuda' and state.get('cuda_rng'):torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(first_epoch,config['epochs']):
        network.train(); losses=[]
        for step,images in enumerate(tqdm(fit_loader,desc=f'Epoch {epoch+1}')):
            if step>=config['max_steps_per_epoch']:break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type,enabled=DEVICE.type=='cuda'):
                loss=denoising_loss(network,images.to(DEVICE),config['diffusion_steps'])
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(),1.)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        val_errors=[]
        for images in val_loader:
            _,residual=reconstruct(network,images.to(DEVICE),config['diffusion_steps'],8,config['seed'])
            val_errors.extend(residual.mean((1,2,3)).cpu().tolist())
        val_loss=statistics.mean(val_errors); improved=val_loss<best
        best=min(best,val_loss);stale=0 if improved else stale+1
        history.append({'epoch':epoch+1,'loss':statistics.mean(losses),'normal_validation_l1':val_loss})
        state={'model':network.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
               'epoch':epoch,'best':best,'history':history,'config':config,
               'training_bases':[r['base_id'] for r in fit_rows],'torch_rng':torch.get_rng_state(),
               'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else None}
        torch.save(state,destination/'last.pt')
        if improved:torch.save(state,destination/'best.pt')
        (destination/'history.json').write_text(json.dumps(history,indent=2))
        print(history[-1])
        if stale>=8:break
    return network,history''','implementation')
    code('''if RUN_TRAINING:
    trained_model,fresh_history=train_normal_model(training_rows,validation_rows,CFG,RUN_DIR/'training')
else:
    print('Fresh training skipped. The next table is the saved historical run, not this session.')''','opt-in')
    md('''## 11. What the saved training run actually did

The run completed eight epochs and selected its best normal-validation checkpoint. The scored validation subset was small; these curves do not establish broad generalization.''')
    code('''training_summary=read_json(ROOT/'outputs/diffusion/training_summary.json')
display(training_summary)
history=read_csv(ROOT/'outputs/diffusion/training_history.csv')
show_table(history)
show_figure('results/figures/training.png')''','saved-evidence')
    code('''checkpoint_path=ROOT/'outputs/diffusion/checkpoints/best.pt'
if not checkpoint_path.exists():
    raise FileNotFoundError('Attach the existing trained project checkpoint to run reconstruction.')
checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
model.load_state_dict(checkpoint['model']);model.eval()
print('Loaded best checkpoint:',checkpoint_path.name)
print('SHA256:',hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())''')
    code('''if RUN_LIVE_RECONSTRUCTION:
    demo_input=example_batch[:1].to(DEVICE)
    if DEVICE.type=='cuda':torch.cuda.synchronize()
    started=time.perf_counter()
    demo_generated,demo_uncertainty=normal_reconstruction(model,demo_input)
    if DEVICE.type=='cuda':torch.cuda.synchronize()
    print('Live normal-training-image reconstruction seconds:',time.perf_counter()-started)
    print('This is a reconstruction demonstration, not a new anomaly benchmark.')
    panel=Image.new('RGB',(4*160,190),'white');draw=ImageDraw.Draw(panel)
    for i,(label,value) in enumerate([('Normal training image',demo_input),('Reconstruction',demo_generated),
                                     ('Absolute residual',(demo_input-demo_generated).abs()),('Seed disagreement',demo_uncertainty)]):
        panel.paste(pil_image(value).resize((150,150)).convert('RGB'),(i*160,0));draw.text((i*160,157),label,fill='black')
    display(panel)
else:
    print('Live reconstruction skipped.')''','live-demo')
    md('''## 12. Build controlled development anomalies

I use six fault families at strengths 0.15, 0.30 and 0.50. Pure local blur is separated from noise. Foreign content is a procedural pattern, not another downloaded dataset. These images never enter generator training. The mask identifies the intended corrupted region; it is not a manually annotated real fault.''')
    code("FAMILIES=('stripes','dead_pixels','missing_patch','local_blur','patch_duplication','foreign_content')")
    definitions('src/generative/stress.py','case_seed','stress')
    code('''development_rows=read_csv(ROOT/'results/development/generative_v6/development_manifest.csv')
assert_clean(development_rows)
assert set(r['base_id'] for r in development_rows).issubset({r['base_id'] for r in exposed})
with Image.open(resolve_image(development_rows[0])) as source:
    normal_demo=source.convert('L').resize((SIZE,SIZE),Image.Resampling.BILINEAR)
panel=Image.new('RGB',(3*220,2*182),'white');draw=ImageDraw.Draw(panel)
for i,family in enumerate(FAMILIES):
    corrupted,mask=stress(normal_demo,family,.30,case_seed(development_rows[0]['base_id'],family))
    x=(i%3)*220;y=(i//3)*182
    panel.paste(corrupted.convert('RGB'),(x,y));panel.paste(mask.resize((64,64)).convert('RGB'),(x+140,y))
    draw.text((x,y+135),family,fill='black')
display(panel)
print('Left: corruption. Right: mask. Severity 0.30; development source only.')''','live-data')
    md('''## 13. Generative discrepancies: pixel and texture

Pixel error is downweighted when the two reconstructions disagree. Texture deficiency compares local Laplacian energy: a blurred observation may contain less high-frequency structure than the generated terrain. Natural smooth terrain can cause false positives, so clean calibration is essential.''')
    definitions('src/generative/discrepancy.py','top_fraction','local_energy','directional_features')
    definitions('scripts/run_generative_development.py','score_maps')
    md('''## 14. Refinement: local residual and narrow frequency peaks

The first three-channel fusion was weak. I tested local residual normalization to suppress broad reconstruction mismatch, and narrow-band spectral contrast to separate periodic peaks from broadband terrain energy. The four-pixel edge exclusion is a limitation: edge faults can be missed.''')
    definitions('scripts/refine_generative_scores.py','frequency_peaks','robust_scale','refined')
    code('''if RUN_LIVE_RECONSTRUCTION:
    demo_scores,demo_maps=score_maps(demo_input,demo_generated,demo_uncertainty)
    item={'input':demo_input[0].cpu(),'reconstruction':demo_generated[0].cpu(),'maps':demo_maps[0].cpu()}
    print('Initial pixel / texture / spectral raw scores:',demo_scores[0].cpu().tolist())
    print('Refined raw scores for the same normal demo:',refined(item).tolist())
    assert torch.isfinite(refined(item)).all()''','live-check')
    md('''## 15. Optional DINO semantic discrepancy — not part of V7 results

This is the proposed corresponding-patch semantic extension. DINOv2 is frozen; cosine differences compare the observation with its generated reconstruction. Top 0.5% aggregation avoids diluting tiny anomalies. It remains **untested in the recorded V7 study** because the local DINO runtime is blocked.

Setting `RUN_DINO=True` in a working environment may download pretrained model weights, not another image dataset. No semantic result should be claimed from this section while the switch is off.''')
    code('''timm=None
load_file=None
if RUN_DINO:
    import timm
    from safetensors.torch import load_file
else:
    print('DINO skipped; V7 contains pixel, texture and directional spectral channels only.')''','opt-in')
    definitions('src/semantic/features.py','FrozenDINO','balanced_memory','cosine_knn')
    definitions('src/generative/discrepancy.py','discrepancy_channels')
    code('''if RUN_DINO:
    dino=FrozenDINO('vit_small_patch14_dinov2.lvd142m').to(DEVICE)
    if not RUN_LIVE_RECONSTRUCTION: raise RuntimeError('Enable live reconstruction for this comparison.')
    with torch.no_grad():
        observed_tokens,grid=dino(demo_input)
        generated_tokens,_=dino(demo_generated)
        four_scores,four_maps=discrepancy_channels(demo_input,demo_generated,demo_uncertainty,observed_tokens,generated_tokens)
    print('Patch grid:',grid,'token shape:',tuple(observed_tokens.shape))
    print('Uncalibrated four-channel normal-demo scores:',four_scores.cpu().tolist())
else:
    print('No DINO embeddings or scores were generated in this run.')''','opt-in')
    md(r'''## 16. Continuous, normal-only calibration

I fit a median and dispersion on calibration A, then compute the threshold from calibration B. A zero MAD is possible for one-sided texture maps; the normal 90th–10th percentile spread supplies a fallback. No anomalous scores determine the threshold.

$$z_j=(s_j-\operatorname{median}_A(s_j))/\operatorname{scale}_A(s_j),\qquad \tau=Q_{0.95}(S_B).$$

A 95th-percentile calibration threshold does not guarantee 5% false positives on new source observations.''')
    code('''def fit_clean_calibration(a_rows,a_scores,b_rows,b_scores,aggregation='mean'):
    assert_clean(a_rows);assert_clean(b_rows)
    assert_group_separation({'calibration_a':a_rows,'calibration_b':b_rows})
    median,scale=robust_scale(a_scores)
    z=(b_scores-median)/scale
    fused=z.mean(1) if aggregation=='mean' else z.max(1).values
    threshold=torch.quantile(fused,.95)
    return {'median':median,'scale':scale,'threshold':float(threshold),'aggregation':aggregation}

def calibrated_scores(scores,state):
    z=(scores-state['median'])/state['scale']
    return z.mean(1) if state['aggregation']=='mean' else z.max(1).values''','implementation')
    code('''v6=read_json(ROOT/'results/development/generative_v6/summary.json')
v7=read_json(ROOT/'results/development/generative_v7/summary.json')
show_table([{'role':'calibration A','images':v7['normal_calibration_a']},
            {'role':'calibration B','images':v7['normal_calibration_b']}])
print('Saved V7 median:',v7['median'])
print('Saved V7 scale:',v7['scale'])
print('Saved V7 mean-fusion threshold:',v7['best_diagnostic']['threshold'])''','saved-evidence')
    md('''## 17. Optional fresh development inference

The full loop is visible below. It uses the same named clean A/B and development manifests, creates synthetic faults only in development, and saves its own predictions. It never loads the final reserve. With the switch off, the notebook reviews the completed experiment in the following sections.''')
    code('''@torch.no_grad()
def score_record(network,row,family='clean',severity=0.):
    with Image.open(resolve_image(row)) as source:
        image=source.convert('L').resize((SIZE,SIZE),Image.Resampling.BILINEAR)
    if family!='clean':image,_=stress(image,family,severity,case_seed(row['base_id'],family))
    x=tensor_image(image)[None].to(DEVICE)
    generated,uncertainty=normal_reconstruction(network,x)
    _,maps=score_maps(x,generated,uncertainty)
    return refined({'input':x[0].cpu(),'reconstruction':generated[0].cpu(),'maps':maps[0].cpu()})

def run_notebook_development(network,destination):
    a=read_csv(ROOT/'results/development/generative_v6/calibration_a_manifest.csv')
    b=read_csv(ROOT/'results/development/generative_v6/calibration_b_manifest.csv')
    dev=development_rows
    assert_group_separation({'a':a,'b':b,'development':dev,'final':parts['final_holdout']})
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    def score_normals(rows,label):
        values=[]
        for i,row in enumerate(tqdm(rows,desc=label)):
            score=score_record(network,row);values.append(score)
            torch.save({'base_id':row['base_id'],'score':score},destination/f'{label}_{i}.pt')
        return torch.stack(values)
    a_scores=score_normals(a,'calibration_a');b_scores=score_normals(b,'calibration_b')
    state=fit_clean_calibration(a,a_scores,b,b_scores)
    torch.save(state,destination/'calibration.pt')
    predictions=[]
    for row in tqdm(dev,desc='Development originals'):
        cases=[('clean',0.)]+[(family,severity) for family in FAMILIES for severity in CFG['severities']]
        for family,severity in cases:
            raw=score_record(network,row,family,severity)
            value=float(calibrated_scores(raw[None],state)[0])
            predictions.append({'base_id':row['base_id'],'source_id':row['source_id'],'family':family,
                'severity':severity,'is_anomaly':int(family!='clean'),'score':value,'threshold':state['threshold']})
        # Persist each completed original so a failure does not erase completed work.
        (destination/'predictions.json').write_text(json.dumps(predictions,indent=2))
    return predictions,state''','implementation')
    code('''if RUN_DEVELOPMENT_INFERENCE:
    fresh_predictions,fresh_calibration=run_notebook_development(model,RUN_DIR/'development')
    print('Fresh development cases:',len(fresh_predictions))
else:
    print('Full development inference skipped. Reviewing the saved V6/V7 experiment below.')''','opt-in')
    md('''## 18. Metrics implemented explicitly

AUROC measures ranking, while FPR/TPR use the clean-calibrated threshold. Average precision depends strongly on anomaly prevalence. Tied scores are handled together. Paired comparison checks whether each corrupted image scores above its own clean source.''')
    definitions('scripts/evaluate_cached_fusion.py','quantile','curves','metrics')
    code('''v7_predictions=[r for r in read_csv(ROOT/'results/development/generative_v7/predictions.csv') if r['variant']=='complete_three_mean']
v7_values=[float(r['score']) for r in v7_predictions]
v7_threshold=float(v7_predictions[0]['threshold'])
recomputed=metrics(v7_predictions,v7_values,v7_threshold)
for key in ['auroc','auprc','f1','precision','recall']:
    assert abs(recomputed[key]-v7['best_diagnostic'][key])<1e-10
show_table([{'metric':key,'value':recomputed[key]} for key in
            ['auroc','auprc','f1','precision','recall','fpr','paired_anomaly_gt_clean']])''','recomputed-evidence')
    code('''show_table([{'actual':'normal','predicted normal':recomputed['tn'],'predicted anomaly':recomputed['fp']},
            {'actual':'anomaly','predicted normal':recomputed['fn'],'predicted anomaly':recomputed['tp']}])
print('Only 16 clean development images: each false positive changes FPR by 6.25 percentage points.')''')
    md('''## 19. Compare V6 and V7 on the same development cases

The refinement improved ranking and recall, but false positives remain too high and the sample is small. It would be misleading to compare this subtle six-family result directly with the older benchmark.''')
    code('''show_table([{'version':'V6 initial maximum',**v6['complete_three_max']},
            {'version':'V7 refined mean',**v7['best_diagnostic']}],
           ['version','auroc','tpr','fpr','f1','paired_anomaly_gt_clean'])
bars(['V6 initial max','V7 refined mean'],[v6['complete_three_max']['auroc'],v7['best_diagnostic']['auroc']],
     'Same-benchmark development AUROC','AUROC')''','saved-evidence')
    code('''v7_ablations=read_csv(ROOT/'results/development/generative_v7/ablations.csv')
show_table(v7_ablations,['variant','auroc','tpr','fpr','f1'])
bars([r['variant'] for r in v7_ablations],[r['auroc'] for r in v7_ablations],
     'V7 ablations — inspected development results','AUROC')''','saved-evidence')
    md('''## 20. Which faults remain difficult?

Every family is evaluated against the same clean references. Multiple corruptions of one original are dependent observations; 288 corruption cases do not mean 288 independent Mars sources.''')
    code('''family_metrics=[r for r in read_csv(ROOT/'results/development/generative_v7/by_family_severity.csv')
                if r['variant']=='complete_three_mean' and r['dimension']=='family']
show_table(family_metrics,['value','auroc','tpr','median_anomaly_score','paired_anomaly_gt_clean'])
bars([r['value'] for r in family_metrics],[r['auroc'] for r in family_metrics],
     'V7 AUROC by corruption family','AUROC')''','saved-evidence')
    code('''severity_metrics=[r for r in read_csv(ROOT/'results/development/generative_v7/by_family_severity.csv')
                  if r['variant']=='complete_three_mean' and r['dimension']=='severity']
show_table(severity_metrics,['value','auroc','tpr','fpr','paired_anomaly_gt_clean'])
bars([r['value'] for r in severity_metrics],[r['auroc'] for r in severity_metrics],
     'V7 AUROC by severity','AUROC')''','saved-evidence')
    md('''## 21. Localization: image score versus pixel map

The image threshold and pixel threshold are different quantities. Pixel normalization uses clean A and the pixel threshold uses clean B. The spectral image score uses frequency features; its row/column map is attribution rather than a spatial inverse FFT. A full-image stripe mask has no negative pixels, so pixel AUROC is undefined.''')
    definitions('scripts/evaluate_generative_localization.py','pixel_auc')
    code('''loc=read_json(ROOT/'results/development/generative_v7/localization_summary.json')
show_table(read_csv(ROOT/'results/development/generative_v7/localization_by_family.csv'),
           ['family','cases','pixel_auroc','iou_at_clean_pixel_q95','positive_map_inside_outside_ratio'])
display(loc['source_bootstrap'])
print('Bootstrap unit: source observation. This interval is conditional on development selection.')''','saved-evidence')
    md('''## 22. Look at successful and unsuccessful explanations

I inspect the normal source, corruption, reconstruction, uncertainty and maps together. Each heatmap has its own display scale. A bright map alone is not evidence of accurate segmentation.''')
    for family in ['dead_pixels','stripes','missing_patch','local_blur','patch_duplication','foreign_content']:
        md(f'### {family.replace("_"," ").capitalize()} — saved severity 0.30 example')
        code(f"show_figure('results/figures/generative_v7/{family}.png')",'saved-evidence')
    code('''false_positives=[r for r in v7_predictions if float(r['is_anomaly'])==0 and float(r['score'])>v7_threshold]
false_negatives=[r for r in v7_predictions if float(r['is_anomaly'])==1 and float(r['score'])<=v7_threshold]
print('False positives:')
show_table(false_positives,['base_id','source_id','score','threshold'])
print('Ten lowest-scoring missed corruptions:')
show_table(sorted(false_negatives,key=lambda r:float(r['score']))[:10],['base_id','family','severity','score','threshold'])''')
    md('''## 23. Historical independent-expert benchmark

For context, the earlier experiment used 128 clean images and 1,056 stronger/different synthetic cases. The 0.9011 equal-weight result belongs to that experiment. The diagnostic DINO/spectral-only combination reached a higher AUROC but was rejected because it excludes diffusion and generated 25% false positives.''')
    code('''old_predictions=read_csv(ROOT/'outputs/evaluation/predictions.csv')
old_reference=read_json(ROOT/'outputs/evaluation/metrics.json')
old_metrics=metrics(old_predictions,[float(r['score']) for r in old_predictions],old_reference['threshold'])
assert abs(old_metrics['auroc']-old_reference['auroc'])<1e-12
show_table(read_csv(ROOT/'outputs/ablation_results.csv'),['variant','auroc','recall','false_positive_rate','f1'])
show_figure('results/figures/baseline_roc.png')
show_figure('results/figures/baseline_pr.png')''','recomputed-evidence')
    md('''## 24. What happens when anomalies are rare?

[VAD4Space](https://arxiv.org/abs/2603.13993) motivates a 5% anomaly setting and efficiency reporting. I use it as related work; I do not import its lunar/rover datasets or geological anomaly labels. Here, prevalence reweighting is analytical sensitivity analysis, not a new test set or deployment result.''')
    definitions('scripts/evaluate_rare_prevalence.py','prevalence_metrics')
    code('''rare_estimate=prevalence_metrics([int(float(r['is_anomaly'])) for r in old_predictions],
                               [float(r['score']) for r in old_predictions],old_reference['threshold'],.05)
display(rare_estimate)
show_table(read_csv(ROOT/'results/development/rare_prevalence/sensitivity.csv'),
           ['variant','assumed_anomaly_prevalence','expected_precision','expected_false_alerts_per_1000'])
print('Assumption: class-conditional score distributions remain unchanged as prevalence changes.')''','recomputed-evidence')
    md('''## 25. Sanity checks before drawing conclusions

These checks execute in this notebook. They cover group separation, mask coverage, tied-score ranking, calibration dispersion and deterministic corruption. They are not a replacement for the full numerical test suite, whose collection is blocked on the original Windows environment.''')
    code('''assert_group_separation(parts)
assert torch.equal(covering_masks(SIZE).sum(0),torch.ones(1,SIZE,SIZE))
assert pixel_auc(torch.tensor([0,1]),torch.tensor([1.,1.]))==.5
assert pixel_auc(torch.tensor([1,1]),torch.tensor([0.,1.])) is None
median,scale=robust_scale(torch.tensor([[0.],[0.],[0.],[0.],[1.],[2.]]))
assert float(scale)>.1
for family in FAMILIES:
    first,first_mask=stress(normal_demo,family,.30,42)
    second,second_mask=stress(normal_demo,family,.30,42)
    assert first.tobytes()==second.tobytes() and first_mask.tobytes()==second_mask.tobytes()
assert not (ROOT/'results/final/final_run_claim.json').exists()
print('Inline sanity checks passed; final evaluation has not been claimed.')''','live-check')
    md('''## 26. Freeze before final evaluation

The final-run guard below is **defined but not called**. The architecture, DINO choice, aggregation, calibration and implementation hashes must be fixed before the one-time final evaluation. Unknown external exposure and the reserve of only three source observations remain limitations.''')
    definitions('src/generative/freeze.py','fingerprint','freeze','claim_final')
    code("print('Final status: NOT RUN. No freeze or final-run claim is created by this notebook.')")
    md('''## 27. Model evolution and conclusions

The model evolved through scoring ablations, a failed calibration experiment, an initial generative discrepancy and a refinement. Some versions reuse one checkpoint; they are not separate training runs. Development scores improved, but false positives and copy-move failures remain unresolved.''')
    code("display(Markdown((ROOT/'MODEL_EVOLUTION.md').read_text(encoding='utf-8')))",'saved-evidence')
    md('''### What I can conclude

- Normal-only masked diffusion runs and can provide useful fault localization.
- Local residual and narrow-band refinements improve this bounded development study.
- Natural terrain variability and copied terrain remain difficult.
- Synthetic benchmarks do not establish real-fault performance.
- The DINO extension, larger validation and untouched final evaluation remain incomplete.

### Reproducibility notes

The inline model, loss, sampling, corruption and scoring definitions are visible above. The exact original run provenance remains in `outputs/` and `results/development/`; fresh notebook runs use separate folders. GPU timing in the live demo is not substituted for historical benchmark timing.

**References:** NASA/JPL HiRISE v3.2, [Zenodo 4002935](https://zenodo.org/records/4002935); Genilotti et al., [VAD4Space](https://arxiv.org/abs/2603.13993); the supplied MarsPS practice statement.''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'TCMD-SS Python','language':'python','name':'tcmdss'},
        'language_info':{'name':'python'},'tcmdss':{'style':'inline Kaggle/Colab walkthrough','default_mode':'review','final_evaluated':False}})
    nb=structure_notebook(nb)
    path=ROOT/'notebooks/TCMD_SS_Final.ipynb'
    archive=ROOT/'notebooks/archive';archive.mkdir(parents=True,exist_ok=True)
    for old in [path,ROOT/'TCMD_SS_HiRISE_Anomaly_Detection.ipynb']:
        if old.exists():
            digest=hashlib.sha256(old.read_bytes()).hexdigest()[:12]
            backup=archive/f'{old.stem}_{digest}.ipynb'
            if not backup.exists():backup.write_bytes(old.read_bytes())
    kernel_dir=ROOT/'outputs/jupyter/kernels/tcmdss';kernel_dir.mkdir(parents=True,exist_ok=True)
    (kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'TCMD-SS Python','language':'python'}))
    runtime=ROOT/'outputs/jupyter/runtime';runtime.mkdir(parents=True,exist_ok=True)
    os.environ['JUPYTER_RUNTIME_DIR']=str(runtime);os.environ['IPYTHONDIR']=str(ROOT/'outputs/jupyter/ipython')
    manager=KernelManager(kernel_name='tcmdss',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
    def progress(cell,cell_index,**kwargs):
        if cell.cell_type=='code':
            print(f'Executing cell {cell_index+1}/{len(nb.cells)}: {cell.source.splitlines()[0][:75]}',flush=True)
    try:
        NotebookClient(nb,timeout=240,km=manager,on_cell_start=progress,resources={'metadata':{'path':str(ROOT)}}).execute()
    except Exception:
        nbf.write(nb,ROOT/'outputs/walkthrough_failed.ipynb')
        raise
    nbf.validate(nb)
    nbf.write(nb,path);nbf.write(nb,ROOT/'TCMD_SS_HiRISE_Anomaly_Detection.ipynb')
    stats={'cells':len(nb.cells),'code_cells':sum(c.cell_type=='code' for c in nb.cells),
           'markdown_cells':sum(c.cell_type=='markdown' for c in nb.cells),
           'execution_errors':sum(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs),
           'fresh_training_executed':False,'full_development_inference_executed':False,'live_reconstruction_executed':True,'final_evaluated':False}
    (ROOT/'results/development/walkthrough_validation.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2));print(path)


if __name__=='__main__':build()
