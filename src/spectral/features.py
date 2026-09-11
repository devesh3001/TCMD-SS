import torch
import torch.nn.functional as F


def window_features(images: torch.Tensor, window: int) -> tuple[torch.Tensor, int]:
    if images.shape[-1] % window:
        images = F.interpolate(images, (images.shape[-1]//window*window,)*2, mode="bilinear", align_corners=False)
    patches = images.unfold(2,window,window).unfold(3,window,window)
    grid = patches.shape[2]
    patches = patches.reshape(len(images), grid*grid, window, window)
    centered = patches-patches.mean((-2,-1),keepdim=True)
    energy = torch.fft.fft2(centered, norm="ortho").abs().square()
    fy = torch.fft.fftfreq(window, device=images.device)
    radial = (fy[:,None].square()+fy[None,:].square()).sqrt()
    bands = [(radial>low)&(radial<=high) for low,high in [(0,.18),(.18,.36),(.36,.72)]]
    features = [torch.log1p(energy[...,band].mean(-1)*100) for band in bands]
    dx = (patches[...,1:]-patches[...,:-1]).square().mean((-2,-1))
    dy = (patches[...,1:,:]-patches[...,:-1,:]).square().mean((-2,-1))
    features += [torch.log1p(dx*100), torch.log1p(dy*100), (dx-dy)/(dx+dy+1e-5),
                 patches.std((-2,-1)), patches.mean((-2,-1))]
    return torch.stack(features, dim=-1), grid


def fit_spectral(images: torch.Tensor, windows=(8,16,32)) -> dict:
    state = {}
    for window in windows:
        features,_ = window_features(images, window)
        flat = features.flatten(0,1)
        center = flat.median(0).values
        scale = (flat-center).abs().median(0).values*1.4826
        state[window] = {"center": center, "scale": scale.clamp_min(.01)}
    return state


def score_spectral(images, state):
    maps = []
    for window, statistics in state.items():
        features, grid = window_features(images, window)
        z = ((features-statistics["center"].to(images.device))/statistics["scale"].to(images.device)).abs().clamp_max(100)
        patch = z.mean(-1).reshape(len(images),1,grid,grid)
        maps.append(F.interpolate(patch, images.shape[-2:], mode="bilinear", align_corners=False))
    heatmap = torch.stack(maps).mean(0)
    score = heatmap.flatten(1).topk(max(1, int(heatmap[0].numel()*.05)), dim=1).values.mean(1)
    return score, heatmap
