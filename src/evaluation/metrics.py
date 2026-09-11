import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score


def metrics(labels, scores, threshold: float) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = scores > threshold
    tn, fp, fn, tp = confusion_matrix(labels,predicted,labels=[0,1]).ravel()
    both = len(np.unique(labels)) == 2
    return {"auroc": float(roc_auc_score(labels,scores)) if both else None,
            "auprc": float(average_precision_score(labels,scores)) if both else None,
            "f1": float(f1_score(labels,predicted,zero_division=0)),
            "precision": float(precision_score(labels,predicted,zero_division=0)),
            "recall": float(recall_score(labels,predicted,zero_division=0)),
            "specificity": float(tn/(tn+fp)) if tn+fp else None,
            "balanced_accuracy": float(balanced_accuracy_score(labels,predicted)) if both else None,
            "false_positive_rate": float(fp/(tn+fp)) if tn+fp else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": len(labels)}


def bootstrap_auc(frame, score_column="score", repeats: int = 200, seed: int = 42):
    groups = [part for _, part in frame.groupby("base_id")]
    rng, values = np.random.default_rng(seed), []
    for _ in range(repeats):
        chosen = rng.integers(0,len(groups),len(groups))
        labels = np.concatenate([groups[i].is_anomaly.to_numpy() for i in chosen])
        scores = np.concatenate([groups[i][score_column].to_numpy() for i in chosen])
        if len(np.unique(labels)) == 2:
            values.append(roc_auc_score(labels,scores))
    return {"lower": float(np.quantile(values,.025)) if values else None,
            "upper": float(np.quantile(values,.975)) if values else None,
            "valid_replicates": len(values), "unit": "source base landmark", "seed": seed}
