"""Content fingerprints and an exclusive final-run claim prevent silent retesting."""
import hashlib
import json
from pathlib import Path


def fingerprint(paths):
    result = {}
    for path in sorted(map(Path, paths)):
        with path.open("rb") as stream:
            result[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def freeze(directory: Path, paths, decisions: dict):
    required = {"architecture", "dino_layers", "top_fraction", "cluster_count", "fusion",
                "sampling_steps", "seeds", "threshold_procedure", "development_complete"}
    if required - decisions.keys() or decisions["development_complete"] is not True:
        raise ValueError("Development must finish and every architecture decision must be explicit")
    directory.mkdir(parents=True, exist_ok=True)
    state = {"decisions": decisions, "sha256": fingerprint(paths)}
    with (directory / "freeze.json").open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)
    return state


def claim_final(directory: Path):
    state = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    if fingerprint(state["sha256"]) != state["sha256"]:
        raise ValueError("Frozen inputs changed; final evaluation is forbidden")
    # Exclusive creation rejects concurrent or repeated final runs, including failed attempts.
    with (directory / "final_run_claim.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "started", "freeze": state}, stream, indent=2)
    return state
