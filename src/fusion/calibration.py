import numpy as np
from src.data.protocol import assert_normal

BRANCHES = ["diffusion", "semantic", "spectral", "latent"]
# Fixed before any corruption benchmark is scored. No metric-guided weight search.
VARIANTS = {"V1_diffusion": [1,0,0,0], "V2_diffusion_semantic": [.4/.65,.25/.65,0,0],
            "V3_TCMD_SS": [.4,.25,.2,.15], "without_semantic": [.4/.75,0,.2/.75,.15/.75],
            "without_spectral": [.4/.8,.25/.8,0,.15/.8], "without_latent": [.4/.85,.25/.85,.2/.85,0],
            "equal_weight": [.25,.25,.25,.25]}


def normalized(scores, state):
    z = (np.asarray(scores)-np.asarray(state["median"]))/np.asarray(state["scale"])
    # Positive deviations are evidence of abnormality; low distances are not penalized.
    return np.clip(z, 0, 100)


def fit_calibration(frame, scores) -> dict:
    assert_normal(frame)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(frame),4) or not np.isfinite(scores).all():
        raise ValueError("Expected finite normal scores for every calibration image")
    median = np.median(scores,axis=0)
    scale = 1.4826*np.median(np.abs(scores-median),axis=0)+1e-6
    state = {"median": median.tolist(), "scale": scale.tolist(), "branches": BRANCHES,
             "normal_count": len(frame), "quantile": .99, "weights": VARIANTS,
             "method": "positive robust z; fixed weights; normal-only 99th percentile"}
    z = normalized(scores,state)
    state["thresholds"] = {name: float(np.quantile(z @ np.array(weights), .99)) for name,weights in VARIANTS.items()}
    return state


def fuse(scores, state, variant="V3_TCMD_SS"):
    return normalized(scores,state) @ np.array(VARIANTS[variant])


def explain(scores, state):
    contributions = normalized(np.asarray(scores),state)*np.array(VARIANTS["V3_TCMD_SS"])
    order = np.argsort(contributions)[::-1]
    descriptions = ["localized diffusion reconstruction failure", "distance from normal semantic patches",
                    "multi-scale frequency/directional deviation", "departure from the normal latent distribution"]
    return "Score driven primarily by " + descriptions[order[0]] + " and " + descriptions[order[1]] + ".", contributions
