import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config
from src.diffusion.train import train

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/smoke.yaml")
    parser.add_argument("--resume")
    args=parser.parse_args()
    config=load_config(args.config)
    train(config, args.resume)
