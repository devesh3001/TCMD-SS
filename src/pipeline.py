import hashlib
import time
import platform
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils.config import ROOT, output_dir, save_json
from src.data.protocol import NormalImages, select_normal, assert_normal, assert_disjoint
from src.diffusion.model import MaskedUNet
from src.diffusion.process import reconstruct
from src.semantic.features import FrozenDINO, semantic_map
from src.spectral.features import fit_spectral, score_spectral
from src.latent.model import fit_latent, score_latent
from src.fusion.calibration import BRANCHES, VARIANTS, fit_calibration, fuse
from src.evaluation.metrics import metrics, bootstrap_auc
from src.evaluation.explanations import render_explanation


def model_hashes(config):
    paths = ["diffusion/checkpoints/best.pt", "semantic/memory.pt", "spectral/normal.pt", "latent/normal.pt"]
    hashes = {}
    for name in paths:
        with (output_dir(config)/name).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream,"sha256").hexdigest()
    return hashes


def checkpoint_hash(config):
    with (output_dir(config)/"diffusion/checkpoints/best.pt").open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def build_memory(config):
    out=output_dir(config)
    frame=pd.read_csv(out/"diffusion/training_manifest.csv",low_memory=False)
    assert_normal(frame)
    device="cuda" if torch.cuda.is_available() else "cpu"
    encoder=FrozenDINO(config["dino_model"],config.get("dino_weights")).to(device)
    loader=DataLoader(NormalImages(frame,config["size"]),batch_size=config["batch_size"])
    quota=max(1,config["memory_patches"]//len(frame))
    memory, global_features=[],[]
    generator=torch.Generator().manual_seed(config["seed"])
    for images in loader:
        tokens,_=encoder(images.to(device)); tokens=tokens.cpu()
        global_features.append(F.normalize(tokens.mean(1),dim=-1))
        for sample in tokens:
            ids=torch.randperm(len(sample),generator=generator)[:quota]
            memory.append(sample[ids])
    (out/"semantic").mkdir(exist_ok=True)
    torch.save({"memory": torch.cat(memory)[:config["memory_patches"]],"global_features": torch.cat(global_features),
                "base_ids": frame.base_id.tolist(), "model": config["dino_model"]},out/"semantic/memory.pt")
    save_json(out/"semantic/metadata.json",{"requested_full_model":"vit_large_patch16_dinov3.sat493m",
        "actual_model":config["dino_model"],"pretrained":True,"frozen":True,
        "fallback_reason":config.get("fallback_reason"),"patches":min(sum(len(x) for x in memory),config["memory_patches"]),
        "normal_bases":len(frame),"sampling":"equal quota per original base, deterministic seed"})


def fit_models(config):
    out=output_dir(config)
    frame=pd.read_csv(out/"diffusion/training_manifest.csv",low_memory=False); assert_normal(frame)
    dataset=NormalImages(frame,config["size"])
    images=torch.stack([dataset[i] for i in range(len(frame))])
    spectral=fit_spectral(images)
    memory=torch.load(out/"semantic/memory.pt",weights_only=False,map_location="cpu")
    latent=fit_latent(memory["global_features"].numpy(),config["latent_dimensions"])
    (out/"spectral").mkdir(exist_ok=True);(out/"latent").mkdir(exist_ok=True)
    torch.save(spectral,out/"spectral/normal.pt");torch.save(latent,out/"latent/normal.pt")


class TCMDSS:
    def __init__(self,config):
        self.config=config; self.out=output_dir(config)
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        checkpoint=torch.load(self.out/"diffusion/checkpoints/best.pt",map_location=self.device,weights_only=False)
        self.diffusion=MaskedUNet(config["base_channels"]).to(self.device)
        self.diffusion.load_state_dict(checkpoint["model"]);self.diffusion.eval()
        self.encoder=FrozenDINO(config["dino_model"],config.get("dino_weights")).to(self.device)
        semantic=torch.load(self.out/"semantic/memory.pt",map_location="cpu",weights_only=False)
        if semantic["model"]!=config["dino_model"]:
            raise ValueError("Semantic memory/backbone mismatch")
        self.memory=semantic["memory"]
        self.spectral=torch.load(self.out/"spectral/normal.pt",map_location="cpu",weights_only=False)
        self.latent=torch.load(self.out/"latent/normal.pt",map_location="cpu",weights_only=False)

    @torch.no_grad()
    def score(self,images):
        images=images.to(self.device)
        reconstruction,diffusion=reconstruct(self.diffusion,images,self.config["diffusion_steps"],self.config["sampling_steps"],self.config["seed"])
        diffusion_score=diffusion.flatten(1).topk(max(1,int(images[0].numel()*.05)),dim=1).values.mean(1)
        tokens,side=self.encoder(images)
        semantic_score,semantic=semantic_map(tokens,side,self.memory,self.config["size"])
        spectral_score,spectral=score_spectral(images,self.spectral)
        latent_score=score_latent(F.normalize(tokens.mean(1),dim=-1).cpu().numpy(),self.latent)
        scores=np.column_stack([diffusion_score.cpu(),semantic_score.cpu(),spectral_score.cpu(),latent_score])
        return scores,reconstruction.cpu().numpy(),{"diffusion":diffusion.cpu().numpy(),"semantic":semantic.cpu().numpy(),"spectral":spectral.cpu().numpy()}


def score_frame(engine,frame):
    loader=DataLoader(NormalImages(frame,engine.config["size"],require_normal=False),batch_size=engine.config["batch_size"])
    arrays=[]
    for images in loader:
        scores,_,_=engine.score(images);arrays.append(scores)
    result=frame.reset_index(drop=True).copy()
    result[BRANCHES]=np.concatenate(arrays)
    return result


def calibrate(config):
    out=output_dir(config); frame=select_normal("calibration_normal",config["calibration_limit"],config["seed"])
    assert_normal(frame)
    scored=score_frame(TCMDSS(config),frame)
    state=fit_calibration(frame,scored[BRANCHES].to_numpy())
    state["checkpoint_sha256"]=checkpoint_hash(config)
    state["model_hashes"]=model_hashes(config)
    state["config"]=config
    (out/"calibration").mkdir(exist_ok=True)
    scored.to_csv(out/"calibration/normal_scores.csv",index=False)
    save_json(out/"calibration/calibration.json",state)
    return state


def evaluate(config):
    import json
    out=output_dir(config)
    state=json.loads((out/"calibration/calibration.json").read_text())
    if state.get("model_hashes")!=model_hashes(config) or state["config"]!=config:
        raise ValueError("Checkpoint/config changed: recalibrate normals before evaluation")
    clean=select_normal("test_clean",config["test_limit"],config["seed"])
    anomalies=pd.read_csv(out/"benchmark/manifest.csv",low_memory=False)
    heldout=pd.read_csv(ROOT/"data/splits/test_clean.csv",low_memory=False)
    if not set(anomalies.base_id).issubset(set(heldout.base_id)) or not anomalies.origin.eq("test_corruption").all():
        raise ValueError("Benchmark must derive exclusively from held-out test bases")
    training=pd.read_csv(out/"diffusion/training_manifest.csv",low_memory=False)
    calibration=pd.read_csv(out/"calibration/normal_scores.csv",low_memory=False)
    assert_disjoint({"training":training,"calibration":calibration,"test":pd.concat([clean,anomalies])})
    clean=clean.assign(family="clean",severity=0,corruption="clean")
    engine=TCMDSS(config)
    started=time.perf_counter()
    predictions=score_frame(engine,pd.concat([clean,anomalies],ignore_index=True))
    predictions["score"]=fuse(predictions[BRANCHES].to_numpy(),state)
    threshold=state["thresholds"]["V3_TCMD_SS"]
    predictions["flagged"]=predictions.score>threshold
    (out/"evaluation").mkdir(exist_ok=True)
    predictions.to_csv(out/"evaluation/predictions.csv",index=False)
    overall=metrics(predictions.is_anomaly,predictions.score,threshold)
    overall["auroc_ci95"]=bootstrap_auc(predictions,repeats=config["bootstrap_repeats"],seed=config["seed"])
    overall.update(threshold=threshold,seconds=time.perf_counter()-started,provisional=config["provisional"])
    save_json(out/"evaluation/metrics.json",overall)
    family_rows=[]
    for family in sorted(anomalies.family.unique()):
        part=predictions.loc[predictions.family.isin(["clean",family])]
        family_rows.append({"family":family,**metrics(part.is_anomaly,part.score,threshold)})
    pd.DataFrame(family_rows).to_csv(out/"evaluation/by_family.csv",index=False)
    severity_rows=[]
    for severity in sorted(anomalies.severity.unique()):
        part=predictions.loc[(predictions.is_anomaly.eq(0))|predictions.severity.eq(severity)]
        severity_rows.append({"severity":severity,**metrics(part.is_anomaly,part.score,threshold)})
    pd.DataFrame(severity_rows).to_csv(out/"evaluation/by_severity.csv",index=False)
    class_rows=[{"class_id":int(c),"class_name":part.class_name.iloc[0],"n":len(part),"false_positives":int(part.flagged.sum()),"false_positive_rate":float(part.flagged.mean())}
                for c,part in predictions.loc[predictions.is_anomaly.eq(0)].groupby("class_id")]
    pd.DataFrame(class_rows).to_csv(out/"evaluation/normal_fpr_by_class.csv",index=False)
    # Save one example per family plus the highest-scored clean case, without changing any fit.
    examples=[part.iloc[0] for _,part in predictions.loc[predictions.is_anomaly.eq(1)].groupby("family")]
    examples.append(predictions.loc[predictions.is_anomaly.eq(0)].sort_values("score",ascending=False).iloc[0])
    explanations=[]
    for index,row in enumerate(examples):
        image=NormalImages(pd.DataFrame([row]),config["size"],require_normal=False)[0][None]
        scores,reconstruction,maps=engine.score(image)
        path=out/"heatmaps"/f"example_{index:02d}.png"
        text=render_explanation(image[0,0].numpy(),reconstruction[0,0],{k:v[0,0] for k,v in maps.items()},scores[0],state,path,f"{row['family']} | score={row['score']:.3f} | flagged={row['flagged']}")
        explanations.append({"image":row.image_path,"figure":str(path.relative_to(ROOT)),"explanation":text,"score":float(row.score),"flagged":bool(row.flagged)})
    save_json(out/"heatmaps/explanations.json",explanations)
    return overall


def ablate(config):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=output_dir(config)
    frame=pd.read_csv(out/"evaluation/predictions.csv",low_memory=False)
    state=json.loads((out/"calibration/calibration.json").read_text())
    results=[]
    for name in VARIANTS:
        scores=fuse(frame[BRANCHES].to_numpy(),state,name)
        results.append({"variant":name,"threshold":state["thresholds"][name],**metrics(frame.is_anomaly,scores,state["thresholds"][name])})
    result=pd.DataFrame(results);result.to_csv(out/"ablation_results.csv",index=False)
    fig,axis=plt.subplots(figsize=(10,4));axis.bar(result.variant,result.auroc,color="#175676")
    axis.set(ylabel="Test-only AUROC",ylim=(0,1),title="TCMD-SS fixed-score ablations (provisional run)")
    axis.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(out/"ablation_plot.png",dpi=160);plt.close(fig)
    return result
