"""Development-only fusion ablations using saved scores and clean source-disjoint A/B.

This analysis needs no model inference. It does not implement or validate the new
observed-versus-generated DINO discrepancy, and never reads final-holdout pixels.
"""
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_revision import read, disjoint, preserve_write

BRANCHES = ["diffusion", "semantic", "spectral", "latent"]
CANDIDATES = {"pixel_only": [0], "DINO_only": [1], "spectral_only": [2],
              "DINO_spectral": [1,2], "pixel_DINO": [0,1], "pixel_spectral": [0,2],
              "pixel_DINO_spectral": [0,1,2], "equal_four": [0,1,2,3]}


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values)-1)*fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def curves(labels, scores):
    positives, negatives = sum(labels), len(labels)-sum(labels)
    if not positives or not negatives:
        raise ValueError("Both classes are needed for ROC/PR")
    pairs = sorted(zip(scores, labels), reverse=True)
    result = [{"fpr":0., "tpr":0., "precision":1., "recall":0.}]
    tp = fp = index = 0
    while index < len(pairs):
        end = index
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            tp += pairs[end][1]
            fp += 1-pairs[end][1]
            end += 1
        result.append({"fpr":fp/negatives, "tpr":tp/positives,
                       "precision":tp/(tp+fp), "recall":tp/positives})
        index = end
    return result


def metrics(rows, scores, threshold):
    labels = [int(float(r["is_anomaly"])) for r in rows]
    curve = curves(labels, scores)
    auc = sum((b["fpr"]-a["fpr"])*(b["tpr"]+a["tpr"])/2 for a,b in zip(curve,curve[1:]))
    ap = sum((b["recall"]-a["recall"])*b["precision"] for a,b in zip(curve,curve[1:]))
    tp=sum(y == 1 and s > threshold for y,s in zip(labels,scores))
    fp=sum(y == 0 and s > threshold for y,s in zip(labels,scores))
    positives=sum(labels); negatives=len(labels)-positives
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/positives
    clean = {r["base_id"]:s for r,s,y in zip(rows,scores,labels) if y == 0}
    paired=[s > clean[r["base_id"]] for r,s,y in zip(rows,scores,labels) if y and r["base_id"] in clean]
    return {"auroc":auc,"auprc":ap,"precision":precision,"recall":recall,
            "f1":2*precision*recall/(precision+recall) if precision+recall else 0.,
            "fpr":fp/negatives,"tpr":recall,"tp":tp,"fp":fp,"tn":negatives-fp,"fn":positives-tp,
            "median_anomaly_score":statistics.median(s for s,y in zip(scores,labels) if y),
            "paired_anomaly_gt_clean":sum(paired)/len(paired) if paired else None,
            "paired_n":len(paired),"n":len(rows),"threshold":threshold}


def write_csv(path, rows):
    import io
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
    writer.writeheader();writer.writerows(rows)
    preserve_write(path, stream.getvalue())


def main():
    directory=ROOT/"results/development/cached_fusion_v5"
    paths=[ROOT/"outputs/calibration/normal_scores.csv", ROOT/"outputs/evaluation/predictions.csv"]
    clean=read(paths[0]); development=read(paths[1])
    split_root=ROOT/"data/splits/generative"
    parts={n:read(split_root/f"{n}.csv") for n in ["train_normal","calibration_a","calibration_b","development","final_holdout"]}
    disjoint(parts)
    if any(r["origin"] != "hirise_clean" or float(r["is_anomaly"]) != 0 for r in clean):
        raise ValueError("Calibration contains non-normal data")
    dev_groups={r["split_group_id"] for r in parts["development"]}
    if any(r["split_group_id"] not in dev_groups for r in development):
        raise ValueError("Cached evaluation falls outside development")
    groups_a={r["split_group_id"] for r in parts["calibration_a"]}
    groups_b={r["split_group_id"] for r in parts["calibration_b"]}
    a=[r for r in clean if r["split_group_id"] in groups_a]
    b=[r for r in clean if r["split_group_id"] in groups_b]
    disjoint({"a":a,"b":b,"development":development})
    median=[statistics.median(float(r[k]) for r in a) for k in BRANCHES]
    scale=[max(1e-6,1.4826*statistics.median(abs(float(r[k])-m) for r in a)) for k,m in zip(BRANCHES,median)]
    def zscores(rows):
        result=[[(float(r[k])-m)/s for k,m,s in zip(BRANCHES,median,scale)] for r in rows]
        if not all(math.isfinite(x) for row in result for x in row):
            raise ValueError("Nonfinite cached score")
        return result
    zb, zd=zscores(b),zscores(development)
    results=[]; predictions=[]; details=[]
    for name, indices in CANDIDATES.items():
        for fusion in ["mean","max"] if len(indices)>1 else ["mean"]:
            aggregate=lambda row: statistics.mean(row[i] for i in indices) if fusion=="mean" else max(row[i] for i in indices)
            threshold=quantile([aggregate(row) for row in zb],.95)
            scores=[aggregate(row) for row in zd]
            variant=f"{name}_{fusion}"
            results.append({"variant":variant, **metrics(development,scores,threshold)})
            for row,score in zip(development,scores):
                predictions.append({"variant":variant,"base_id":row["base_id"],"source_id":row["source_id"],
                                    "is_anomaly":int(float(row["is_anomaly"])),"family":row["family"],
                                    "severity":row["severity"],"score":score,"threshold":threshold})
            for dimension in ["family","severity"]:
                for value in sorted({r[dimension] for r in development if int(float(r["is_anomaly"]))}):
                    select=[i for i,r in enumerate(development) if not int(float(r["is_anomaly"])) or r[dimension]==value]
                    details.append({"variant":variant,"dimension":dimension,"value":value,
                                    **metrics([development[i] for i in select],[scores[i] for i in select],threshold)})
    best=max(results,key=lambda r:r["auroc"])
    state={"status":"development_only", "protocol":"Cached independent-expert baseline, NOT generative discrepancy",
           "normal_calibration_a":len(a),"normal_calibration_b":len(b),"median":median,"scale":scale,
           "threshold_procedure":"95th percentile on clean B after clean-A robust normalization",
           "selection":"Highest development AUROC among 13 prespecified discrete ablations; selection bias applies",
           "best_development":best,"final_evaluated":False,
           "limitations":["Historical 11 corruption kinds at .2/.5/.8, not the requested six-family .15/.30/.50 benchmark",
                          "Small clean A/B calibration samples; empirical q95 does not guarantee 5% out-of-source FPR",
                          "No new inference or training; no evidence for improved blur or copy-move generative discrepancy"],
           "input_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    write_csv(directory/"ablations.csv",results)
    write_csv(directory/"predictions.csv",predictions)
    write_csv(directory/"by_family_severity.csv",details)
    preserve_write(directory/"calibration_and_selection.json",json.dumps(state,indent=2)+"\n")
    print(json.dumps(state,indent=2))


if __name__ == "__main__":
    main()
