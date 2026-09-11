import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.utils.config import load_config,output_dir,save_json
from src.data.protocol import NormalImages
from src.pipeline import TCMDSS,model_hashes
from src.fusion.calibration import fuse,BRANCHES
from src.evaluation.explanations import render_explanation

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/local.yaml")
    parser.add_argument("--image",required=True);parser.add_argument("--output",default="outputs/prediction")
    args=parser.parse_args();config=load_config(args.config)
    state=json.loads((output_dir(config)/"calibration/calibration.json").read_text())
    if state["model_hashes"]!=model_hashes(config) or state["config"]!=config:
        raise ValueError("Model/config changed: recalibrate normals first")
    image=NormalImages(pd.DataFrame({"image_path":[args.image]}),config["size"],require_normal=False)[0][None]
    scores,reconstruction,maps=TCMDSS(config).score(image)
    score=float(fuse(scores,state)[0]);threshold=state["thresholds"]["V3_TCMD_SS"]
    path=Path(args.output);path.mkdir(parents=True,exist_ok=True)
    explanation=render_explanation(image[0,0].numpy(),reconstruction[0,0],{k:v[0,0] for k,v in maps.items()},scores[0],state,path/"explanation.png",f"Score {score:.4f}; threshold {threshold:.4f}")
    result={"score":score,"threshold":threshold,"flagged":score>threshold,"raw_branches":dict(zip(BRANCHES,scores[0].tolist())),"explanation":explanation}
    save_json(path/"prediction.json",result);print(result)
