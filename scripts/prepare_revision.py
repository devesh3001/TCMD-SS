"""Reserve unseen source groups without loading images or modifying old experiments."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("source_id", "base_id", "group_id", "split_group_id")


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def disjoint(parts):
    for key in KEYS:
        seen = set()
        for name, rows in parts.items():
            values = {row[key] for row in rows}
            if not values or "" in values or seen & values:
                raise ValueError(f"Empty, missing or overlapping {key}: {name}")
            seen |= values


def preserve_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Preserving existing artifact: {path}")
    else:
        path.write_text(text, encoding="utf-8")


def main():
    import io
    original = {name: read(ROOT / "data/splits" / f"{name}.csv")
                for name in ("train_normal", "calibration_normal", "test_clean")}
    disjoint(original)
    predictions = sorted((ROOT / "outputs").glob("**/predictions.csv"))
    consumed = {key: set() for key in KEYS}
    for path in predictions:
        for row in read(path):
            for key in KEYS:
                consumed[key].add(row[key])
    audit = json.loads((ROOT / "outputs/dataset_audit.json").read_text())
    # Conservatively exclude all audit review components, including linked sources.
    reviewed = {item["group_id"] for item in audit["possible_leakage"]}
    for row in original["test_clean"]:
        if row["group_id"] in reviewed:
            for key in KEYS:
                consumed[key].add(row[key])
    test = original["test_clean"]
    excluded_groups = {r["split_group_id"] for r in test
                       if any(r[k] in consumed[k] for k in KEYS)}
    final = [r for r in test if r["split_group_id"] not in excluded_groups]
    development = [r for r in test if r["split_group_id"] in excluded_groups]
    calibration = original["calibration_normal"]
    groups = sorted({r["split_group_id"] for r in calibration},
                    key=lambda s: hashlib.sha256(("42" + s).encode()).hexdigest())
    half = set(groups[:len(groups)//2])
    parts = {"train_normal": original["train_normal"],
             "calibration_a": [r for r in calibration if r["split_group_id"] in half],
             "calibration_b": [r for r in calibration if r["split_group_id"] not in half],
             "development": development, "final_holdout": final}
    disjoint(parts)
    counts = {}
    for name, rows in parts.items():
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        preserve_write(ROOT / "data/splits/generative" / f"{name}.csv", stream.getvalue())
        counts[name] = {"images": len(rows), **{key: len({r[key] for r in rows}) for key in KEYS}}
    report = {"seed": 42, "counts": counts, "overlap": 0,
              "exposure_files": [str(p.relative_to(ROOT)) for p in predictions],
              "audit_review_components_excluded": sorted(reviewed),
              "normal_classes": list(range(8)), "final_evaluated": False,
              "limitations": ["Exposure history covers saved workspace predictions and audit review components only; externally evaluated image IDs were not supplied.",
                              "Final holdout has few observations and cannot support broad Mars generalization claims.",
                              "Calibration A/B are source-disjoint; previous baseline calibration remains development history."]}
    preserve_write(ROOT / "results/development/split_protocol.json", json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
