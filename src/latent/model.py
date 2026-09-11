import numpy as np


def fit_latent(features: np.ndarray, maximum: int = 32, shrinkage: float = .1) -> dict:
    if len(features) < 3 or not 0 < shrinkage <= 1:
        raise ValueError('Need >=3 normal embeddings and positive shrinkage')
    mean = features.mean(0)
    _, _, vectors = np.linalg.svd(features-mean, full_matrices=False)
    components = vectors[:min(maximum,len(features)-1,features.shape[1])]
    values = (features-mean) @ components.T
    location = values.mean(0)
    covariance = np.atleast_2d(np.cov(values,rowvar=False))
    # Fixed isotropic shrinkage stabilizes small normal-only samples without label tuning.
    target = max(float(np.trace(covariance)/len(covariance)),1e-8)
    covariance = (1-shrinkage)*covariance+shrinkage*target*np.eye(len(covariance))
    return {'mean':mean,'components':components,'location':location,'precision':np.linalg.pinv(covariance),'shrinkage':shrinkage}


def score_latent(features: np.ndarray, state: dict) -> np.ndarray:
    latent = (features-state['mean']) @ state['components'].T-state['location']
    return np.sqrt(np.maximum(np.einsum('bi,ij,bj->b',latent,state['precision'],latent),0))
