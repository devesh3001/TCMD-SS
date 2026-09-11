import time
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.protocol import NormalImages, select_normal, assert_disjoint
from src.diffusion.model import MaskedUNet
from src.diffusion.process import denoising_loss, reconstruct
from src.utils.config import output_dir, save_json
from src.utils.seed import seed_everything


def train(config: dict, resume: str | None = None):
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = output_dir(config)/"diffusion"
    (out/"checkpoints").mkdir(parents=True, exist_ok=True)
    frame = select_normal("train_normal", config["train_limit"], config["seed"])
    # Validation sources come only from the training partition, leaving calibration untouched.
    groups = sorted(frame.split_group_id.unique())
    validation_groups = set()
    for group in groups[::2] + groups[1::2]:
        proposed = validation_groups | {group}
        if set(frame.loc[~frame.split_group_id.isin(proposed), "class_id"]) == set(frame.class_id):
            validation_groups = proposed
        if len(validation_groups) >= max(1,len(groups)//8):
            break
    validation = frame.loc[frame.split_group_id.isin(validation_groups)]
    training = frame.loc[~frame.split_group_id.isin(validation_groups)]
    if training.empty or validation.empty:
        raise ValueError("Need multiple training source groups for normal-only model selection")
    assert_disjoint({"fit": training, "validation": validation})
    training.to_csv(out/"training_manifest.csv", index=False)
    validation.to_csv(out/"validation_manifest.csv", index=False)
    train_loader = DataLoader(NormalImages(training, config["size"]), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(NormalImages(validation.head(config["validation_limit"]), config["size"]), batch_size=config["batch_size"])
    model = MaskedUNet(config["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    start, best, stale, history = 0, float("inf"), 0, []
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        for key in ["size", "base_channels", "diffusion_steps"]:
            if state["config"][key] != config[key]:
                raise ValueError(f"Resume architecture mismatch: {key}")
        if state.get("training_bases", training.base_id.tolist()) != training.base_id.tolist():
            raise ValueError("Resume training subset changed")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, best, stale, history = state["epoch"]+1, state["best"], state["stale"], state["history"]
        torch.set_rng_state(state["torch_rng"].cpu()); random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        if device.type=="cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    started = time.perf_counter()
    for epoch in range(start, config["epochs"]):
        model.train(); losses = []; samples = 0
        for step, images in enumerate(train_loader):
            if config.get("max_steps_per_epoch") and step >= config["max_steps_per_epoch"]:
                break
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type=="cuda"):
                loss = denoising_loss(model, images, config["diffusion_steps"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach())); samples += len(images)
        validation_errors = []
        for images in val_loader:
            _, error = reconstruct(model, images.to(device), config["diffusion_steps"], config["sampling_steps"], config["seed"])
            validation_errors.extend(error.mean((1,2,3)).cpu().tolist())
        value = float(np.mean(validation_errors))
        improved = value < best
        best, stale = (value, 0) if improved else (best, stale+1)
        history.append({"epoch": epoch+1, "steps": len(losses), "samples": samples, "loss": float(np.mean(losses)), "normal_validation_l1": value,
                        "session_seconds": time.perf_counter()-started})
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "training_bases": training.base_id.tolist(), "epoch": epoch, "best": best, "stale": stale, "config": config, "history": history,
                 "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type=="cuda" else None}
        torch.save(state, out/"checkpoints/last.pt")
        if improved:
            torch.save(state, out/"checkpoints/best.pt")
        pd.DataFrame(history).to_csv(out/"training_history.csv", index=False)
        print(f"epoch={epoch+1} loss={history[-1]['loss']:.5f} normal_validation_l1={value:.5f}", flush=True)
        if stale >= config["patience"]:
            break
    save_json(out/"training_summary.json", {"epochs_completed": len(history), "training_images": len(training), "validation_images": len(validation),
               "steps_completed": sum(row["steps"] for row in history), "optimizer_samples_processed": sum(row.get("samples",0) for row in history),
               "validation_scored_images": min(len(validation),config["validation_limit"]), "best_normal_validation_l1": best,
               "device": str(device), "parameters": sum(p.numel() for p in model.parameters()), "session_seconds": time.perf_counter()-started,
               "resumed_from": resume, "provisional": config["provisional"]})
    return model
