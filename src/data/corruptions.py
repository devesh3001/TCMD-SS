import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from src.data.protocol import select_normal
from src.utils.config import output_dir

FAMILIES = {"stripes": "stripes", "dead_lines": "dead lines/pixels", "dead_pixels": "dead lines/pixels",
            "clipping": "clipping", "banding": "stripes", "blur_noise": "blur/noise",
            "missing_crop": "missing crop", "block_permutation": "block corruption",
            "patch_duplication": "patch duplication", "tearing": "block corruption", "foreign": "foreign/synthetic contamination"}


def corrupt(image: np.ndarray, kind: str, severity: float, seed: int,
            foreign_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if kind not in FAMILIES or not 0 < severity <= 1:
        raise ValueError("Unknown corruption or invalid severity")
    rng = np.random.default_rng(seed)
    x = image.astype(np.float32).copy()
    h, w = x.shape
    side = max(4, int(min(h, w) * (0.15 + 0.4 * severity)))
    y, z = int(rng.integers(0, h - side + 1)), int(rng.integers(0, w - side + 1))
    region = np.s_[y:y+side, z:z+side]
    if kind in {"stripes", "banding"}:
        wave = np.sin(np.arange(h) * 2 * np.pi / (4 + int(15 * severity)))[:, None]
        if seed % 2:
            wave = wave.T
        x += (0.15 + 0.35 * severity) * wave
    elif kind == "dead_lines":
        indices = rng.choice(min(h, w), max(1, int(8 * severity)), replace=False)
        if seed % 2:
            x[:, indices] = 0
        else:
            x[indices, :] = 0
    elif kind == "dead_pixels":
        mask = rng.random(x.shape) < 0.01 + 0.12 * severity
        x[mask] = rng.integers(0, 2, mask.sum())
    elif kind == "clipping":
        x = np.clip((x - 0.5) * (1 + 6 * severity) + 0.5, 0, 1)
    elif kind == "blur_noise":
        blurred = np.asarray(Image.fromarray((x*255).astype("uint8")).filter(ImageFilter.GaussianBlur(1+5*severity))) / 255
        x[region] = blurred[region] + rng.normal(0, 0.15*severity, (side, side))
    elif kind == "missing_crop":
        x[region] = 0 if seed % 2 else 1
    elif kind == "block_permutation":
        patch = x[region].copy()
        x[region] = np.roll(patch, side//2, axis=seed % 2)
    elif kind == "patch_duplication":
        x[region] = x[:side, :side]
    elif kind == "tearing":
        x[y:y+side] = np.roll(x[y:y+side], max(2, int(w*severity/3)), axis=1)
        x[y:y+2] = 0
    elif kind == "foreign":
        if foreign_path:
            with Image.open(foreign_path) as source:
                patch = np.asarray(source.convert("L").resize((side, side)), dtype=np.float32)/255
        else:
            yy, xx = np.indices((side, side))
            patch = ((xx//3 + yy//3) % 2).astype(np.float32)
        x[region] = patch
    x = np.clip(x, 0, 1)
    return x, (np.abs(x-image) > 1/255).astype(np.uint8)


def create_benchmark(config: dict) -> pd.DataFrame:
    sources = select_normal("test_clean", config["anomaly_sources"], config["seed"])
    directory = output_dir(config) / "benchmark"
    directory.mkdir(exist_ok=True)
    records = []
    for source in sources.to_dict("records"):
        with Image.open(source["image_path"]) as image:
            values = np.asarray(image.convert("L").resize((config["size"], config["size"])), dtype=np.float32)/255
        for kind, family in FAMILIES.items():
            for severity in config["severities"]:
                identifier = f'{source["base_id"]}_{kind}_{severity}_{config["seed"]}'
                seed = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)
                for retry in range(16):
                    changed, mask = corrupt(values, kind, severity, seed+retry, config.get("foreign_path"))
                    if mask.any():
                        seed += retry
                        break
                else:
                    raise ValueError(f"No observable corruption for {identifier}")
                path = directory / f"{identifier}.png"
                mask_path = directory / f"{identifier}_mask.png"
                Image.fromarray((changed*255).round().astype("uint8")).save(path)
                Image.fromarray(mask*255).save(mask_path)
                records.append({**source, "source_image": source["image_path"], "image_path": str(path.resolve()),
                                "mask_path": str(mask_path.resolve()), "origin": "test_corruption", "is_anomaly": 1,
                                "changed_fraction": float(mask.mean()), "corruption": kind, "family": family, "severity": severity, "corruption_seed": seed})
    frame = pd.DataFrame(records)
    frame.to_csv(directory / "manifest.csv", index=False)
    return frame
