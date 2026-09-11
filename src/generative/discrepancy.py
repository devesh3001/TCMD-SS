"""Torch-only discrepancy channels, keeping generation central to each channel."""
import math
import torch
import torch.nn.functional as F
from src.diffusion.process import reconstruct


def top_fraction(values: torch.Tensor, fraction: float = .005) -> torch.Tensor:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    flat = values.flatten(1)
    return flat.topk(max(1, math.ceil(flat.shape[1]*fraction)), dim=1).values.mean(1)


@torch.no_grad()
def normal_reconstruction(model, images, diffusion_steps=100, sampling_steps=15,
                          seeds=(42, 1042)):
    if images.shape[-2:] != (128, 128):
        raise ValueError("This protocol requires 128x128 and 32x32 tiles")
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("Two distinct deterministic seeds are required")
    if not 15 <= sampling_steps <= 30 or sampling_steps > diffusion_steps:
        raise ValueError("Use 15–30 DDIM steps within the diffusion schedule")
    samples = torch.stack([reconstruct(model, images, diffusion_steps, sampling_steps, s)[0]
                           for s in seeds])
    # Seed disagreement measures generative instability, not calibrated uncertainty.
    return samples.mean(0), samples.std(0, unbiased=False)


def local_energy(images):
    kernel = images.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None]
    laplacian = F.conv2d(F.pad(images, (1, 1, 1, 1), mode="reflect"), kernel)
    return F.avg_pool2d(laplacian.square(), 9, stride=1, padding=4)


def directional_features(images):
    profiles = torch.stack([images.mean(-1), images.mean(-2)], dim=2).squeeze(1)
    profiles = profiles-profiles.mean(-1, keepdim=True)
    window = torch.hann_window(profiles.shape[-1], device=images.device)
    # A tapered global profile emphasizes periodic scan artifacts without local FFT tiles.
    power = torch.fft.rfft(profiles*window, norm="ortho").abs().square()
    return torch.log1p(power[..., 1:]*100)


def discrepancy_channels(observed, generated, uncertainty, observed_tokens, generated_tokens,
                         fraction=.005):
    if observed.shape != generated.shape or observed.shape != uncertainty.shape:
        raise ValueError("Image, reconstruction and uncertainty shapes must match")
    if observed_tokens.shape != generated_tokens.shape:
        raise ValueError("Corresponding DINO grids must match")
    pixel = (observed-generated).abs()/(1+uncertainty*4)
    # A one-sided ratio targets missing texture; clean-only calibration handles terrain variability.
    texture = (torch.log(local_energy(generated)+1e-4)-torch.log(local_energy(observed)+1e-4)).clamp_min(0)
    patch = (1-(F.normalize(observed_tokens, dim=-1)*F.normalize(generated_tokens, dim=-1)).sum(-1)).clamp_min(0)
    grid = math.isqrt(patch.shape[1])
    if grid*grid != patch.shape[1]:
        raise ValueError("DINO patch grid must be square")
    semantic = F.interpolate(patch[:, None].reshape(-1, 1, grid, grid), observed.shape[-2:], mode="bilinear", align_corners=False)
    spectral_difference = (directional_features(observed)-directional_features(generated)).abs()
    # Profile differences localize rows/columns; this is attribution, not frequency inversion.
    row = (observed.mean(-1)-generated.mean(-1)).abs()[..., None]
    column = (observed.mean(-2)-generated.mean(-2)).abs()[:, :, None, :]
    spectral_map = (row+column)/2
    scores = torch.stack([top_fraction(pixel, fraction), top_fraction(texture, fraction),
                          top_fraction(patch, fraction), top_fraction(spectral_difference, fraction)], 1)
    return scores, torch.cat([pixel, texture, semantic, spectral_map], 1)


def fit_robust(clean_a, *, clean_only: bool):
    if not clean_only or clean_a.ndim != 2 or len(clean_a) < 2 or not torch.isfinite(clean_a).all():
        raise ValueError("Finite clean calibration A scores are required")
    median = clean_a.median(0).values
    mad = (clean_a-median).abs().median(0).values
    return median, (1.4826*mad).clamp_min(1e-6)


def fused_score(scores, median, scale, channels=(0, 1, 2, 3)):
    # Continuous z scores avoid empirical-percentile saturation on unseen anomalies.
    return ((scores-median)/scale)[:, list(channels)].amax(1)


def clean_threshold(clean_b, median, scale, *, clean_only: bool, channels=(0, 1, 2, 3)):
    if not clean_only or len(clean_b) < 2 or not torch.isfinite(clean_b).all():
        raise ValueError("Independent finite clean calibration B scores are required")
    return torch.quantile(fused_score(clean_b, median, scale, channels), .95)
