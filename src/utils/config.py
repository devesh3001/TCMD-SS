from pathlib import Path
import json
import yaml
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_config(path="configs/smoke.yaml") -> dict:
    config = yaml.safe_load((ROOT / path).read_text())
    torch.set_num_threads(config.get("threads", 4))
    return config


def output_dir(config: dict) -> Path:
    path = ROOT / config["output"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
