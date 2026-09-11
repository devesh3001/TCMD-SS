from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.fusion.calibration import BRANCHES, explain, VARIANTS


def render_explanation(image, reconstruction, maps, scores, calibration, path: Path, title: str):
    text, contributions = explain(scores,calibration)
    shape = image.shape
    calibrated = []
    for i, branch in enumerate(BRANCHES):
        raw = maps[branch] if branch in maps else np.full(shape,scores[i])
        calibrated.append(np.clip((raw-calibration["median"][i])/calibration["scale"][i],0,100))
    fused = np.stack(calibrated,axis=-1) @ np.array(VARIANTS["V3_TCMD_SS"])
    fig, axes = plt.subplots(2,4,figsize=(14,7))
    items = [(image,"Original", "gray"),(reconstruction,"Diffusion reconstruction","gray"),
             (maps["diffusion"],"Diffusion L1 residual","magma"),(maps["semantic"],"Semantic kNN distance","magma"),
             (maps["spectral"],"Spectral deviation","magma"),(fused,"Fused map (latent is global)","magma")]
    for axis,(values,label,cmap) in zip(axes.flat,items):
        artist=axis.imshow(values,cmap=cmap); axis.set_title(label); axis.axis("off")
        if cmap!="gray": fig.colorbar(artist,ax=axis,fraction=.046,pad=.04)
    axes[1,2].imshow(image,cmap="gray"); axes[1,2].imshow(fused,cmap="magma",alpha=.55)
    axes[1,2].set_title("Anomaly overlay"); axes[1,2].axis("off")
    axes[1,3].bar(BRANCHES,contributions,color=["#175676","#4ba3c3","#ef8354","#59656f"])
    axes[1,3].tick_params(axis="x",rotation=25); axes[1,3].set_title("Actual weighted contributions")
    fig.suptitle(title+"\n"+text,fontsize=11)
    fig.tight_layout(rect=(0,0,1,.9)); path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=140); plt.close(fig)
    np.savez_compressed(path.with_suffix(".npz"), original=image,reconstruction=reconstruction,fused=fused,**maps)
    return text
