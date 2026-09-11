"""A clean training-image GPU smoke; never reads the final holdout."""
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from src.diffusion.model import MaskedUNet
from src.generative.discrepancy import normal_reconstruction


def main():
    directory = ROOT / "results/development/reconstruction_probe"
    if (directory / "metrics.json").exists():
        print((directory / "metrics.json").read_text())
        return  # Existing measurements are evidence and must not be overwritten.
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with (ROOT / "outputs/diffusion/training_manifest.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    with Image.open(row["image_path"]) as source:
        image = source.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        values = torch.tensor(list(image.get_flattened_data()), dtype=torch.float32)
    values = values.reshape(1,1,128,128).to(device)/255
    checkpoint = torch.load(ROOT / "outputs/diffusion/checkpoints/best.pt", map_location="cpu", weights_only=False)
    model = MaskedUNet(32).to(device)
    model.load_state_dict(checkpoint["model"])
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    generated, uncertainty = normal_reconstruction(model, values)
    if device == "cuda":
        torch.cuda.synchronize()
    report = {"purpose": "clean training-image reconstruction smoke, not anomaly performance",
              "base_id": row["base_id"], "source_id": row["source_id"],
              "seeds": [42,1042], "sampling_steps": 15, "seconds": time.perf_counter()-start,
              "mean_absolute_residual": float((values-generated).abs().mean()),
              "mean_seed_disagreement": float(uncertainty.mean()),
              "finite": bool(torch.isfinite(generated).all() and torch.isfinite(uncertainty).all()),
              "gpu_peak_mib": torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else None}
    directory.mkdir(parents=True, exist_ok=True)
    for name, tensor in [("normal", values), ("reconstruction", generated), ("seed_disagreement", uncertainty)]:
        output = Image.new("L", (128,128))
        output.putdata(tensor.clamp(0,1).mul(255).round().to(torch.uint8).cpu().flatten().tolist())
        output.save(directory/f"{name}.png")
    (directory/"metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
