import ast
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.utils.config import ROOT, save_json

GROUP_KEYS = ["base_id", "source_id", "group_id", "split_group_id"]


def assert_disjoint(frames: dict[str, pd.DataFrame]) -> None:
    for key in GROUP_KEYS:
        seen = set()
        for name, frame in frames.items():
            if frame[key].isna().any():
                raise ValueError(f"Missing {key} in {name}")
            values = set(frame[key])
            if values & seen:
                raise ValueError(f"Forbidden {key} overlap in {name}")
            seen.update(values)


def assert_normal(frame: pd.DataFrame) -> None:
    if frame.empty or not frame["is_anomaly"].eq(0).all() or not frame["origin"].eq("hirise_clean").all():
        raise ValueError("Only clean HiRISE records may enter fitting/calibration")
    if "image_path" in frame:
        if not all(Path(path).resolve().is_relative_to((ROOT / "data/raw/hirise_v3_2").resolve()) for path in frame.image_path):
            raise ValueError("Normal fitting paths must remain inside the audited raw dataset")
    if not frame["class_id"].isin(range(8)).all():
        raise ValueError("Unknown semantic labels require review")


def create_protocol() -> dict:
    audit = ROOT / "outputs/dataset_audit.csv"
    frame = pd.read_csv(audit, low_memory=False)
    frame = frame.loc[frame.status.eq("ok")].copy()
    frame["is_original"] = frame.augmentations.map(lambda x: len(ast.literal_eval(x)) == 0)
    frame["origin"], frame["is_anomaly"] = "hirise_clean", 0
    frame["label"] = "normal"
    frame["image_path"] = frame.path.map(lambda p: str((ROOT / "data/raw/hirise_v3_2" / p).resolve()))
    assignments = {"train_normal": "train", "calibration_normal": "val", "test_clean": "test"}
    full = {key: frame.loc[frame.assigned_split.eq(value)].copy() for key, value in assignments.items()}
    assert_disjoint(full)
    directory = ROOT / "data/splits"
    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, part in full.items():
        if name != "train_normal":
            part = part.loc[part.is_original]
        assert_normal(part)
        part.to_csv(directory / f"{name}.csv", index=False)
        counts[name] = len(part)
    report = {"counts": counts, "normal_classes": list(range(8)), "forbidden_overlap": 0,
              "audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
              "policy": "Reuse audited allocation; all classes normal; calibration/test originals only"}
    save_json(ROOT / "outputs/splits/protocol.json", report)
    return report


def select_normal(name: str, limit: int | None, seed: int = 42) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / f"data/splits/{name}.csv", low_memory=False)
    assert_normal(frame)
    # One original per base prevents augmentation-rich landmarks dominating small runs.
    originals = frame.loc[frame.is_original].drop_duplicates("base_id")
    if limit is None or limit >= len(originals):
        return originals.reset_index(drop=True)
    shuffled = [part.sample(frac=1, random_state=seed).to_dict("records") for _, part in originals.groupby("class_id")]
    rows = []
    while len(rows) < limit and any(shuffled):
        for group in shuffled:
            if group and len(rows) < limit:
                rows.append(group.pop())
    return pd.DataFrame(rows).reset_index(drop=True)


class NormalImages(Dataset):
    def __init__(self, frame: pd.DataFrame, size: int, require_normal: bool = True):
        if require_normal:
            assert_normal(frame)
        self.frame, self.size = frame.reset_index(drop=True), size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        with Image.open(self.frame.iloc[index].image_path) as image:
            values = np.asarray(image.convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR), dtype=np.float32).copy() / 255
        return torch.from_numpy(values)[None]
