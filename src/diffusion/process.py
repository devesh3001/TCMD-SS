import math
import torch
import torch.nn.functional as F


def training_masks(batch: int, size: int, device="cpu", generator=None) -> torch.Tensor:
    masks = []
    yy, xx = torch.meshgrid(torch.linspace(0, 1, size, device=device), torch.linspace(0, 1, size, device=device), indexing="ij")
    for _ in range(batch):
        fraction = float(torch.rand((), device=device, generator=generator)*0.3+0.1)
        kind = int(torch.randint(0, 3, (), device=device, generator=generator))
        if kind == 2:
            field = F.interpolate(torch.rand(1, 1, 8, 8, device=device, generator=generator), (size, size), mode="bicubic", align_corners=False)[0, 0]
        else:
            field = torch.full((size, size), -100., device=device)
            for _ in range(1 if kind == 0 else 3):
                center = torch.rand(2, device=device, generator=generator)
                aspect = float(torch.rand((), device=device, generator=generator)+0.5)
                distance = torch.maximum((yy-center[0]).abs()*aspect, (xx-center[1]).abs()/aspect)
                field = torch.maximum(field, -distance)
        mask = torch.zeros(size*size, device=device)
        mask[field.flatten().topk(max(1, int(fraction*size*size))).indices] = 1
        masks.append(mask.reshape(1, size, size))
    return torch.stack(masks)


def covering_masks(size: int, device="cpu") -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
    labels = ((yy // max(1, size//4)) % 2)*2 + (xx // max(1, size//4)) % 2
    return torch.stack([(labels == i).float()[None] for i in range(4)])


def schedule(steps: int, device="cpu") -> torch.Tensor:
    times = torch.linspace(0, steps, steps+1, device=device)/steps
    cumulative = torch.cos((times+0.008)/1.008 * math.pi/2).square()
    cumulative = cumulative / cumulative[0]
    betas = (1-cumulative[1:]/cumulative[:-1]).clamp(0.0001, 0.999)
    return torch.cumprod(1-betas, dim=0)


def denoising_loss(model, images, steps: int):
    x = images*2-1
    masks = training_masks(len(x), x.shape[-1], x.device)
    t = torch.randint(0, steps, (len(x),), device=x.device)
    alpha = schedule(steps, x.device)[t, None, None, None]
    noise = torch.randn_like(x)
    noisy = alpha.sqrt()*x+(1-alpha).sqrt()*noise
    prediction = model(noisy, x*(1-masks), masks, t)
    return ((prediction-noise).square()*masks).sum()/masks.sum().clamp_min(1)


@torch.no_grad()
def reconstruct(model, images, steps: int = 100, sampling_steps: int = 8, seed: int = 42):
    model.eval()
    x = images*2-1
    alphas = schedule(steps, x.device)
    times = torch.linspace(steps-1, 0, sampling_steps, device=x.device).long().unique_consecutive()
    restored = torch.zeros_like(x)
    for index, hidden in enumerate(covering_masks(x.shape[-1], x.device)):
        mask = hidden[None].expand_as(x)
        visible = x*(1-mask)
        generator = torch.Generator(device=x.device).manual_seed(seed+index)
        # Shared image-independent noise makes scores independent of batch position.
        sample = torch.randn((1, 1, *x.shape[-2:]), generator=generator, device=x.device).expand_as(x).clone()
        for position, timestep in enumerate(times):
            t = timestep.expand(len(x))
            eps = model(sample, visible, mask, t)
            alpha = alphas[timestep]
            estimate = ((sample-(1-alpha).sqrt()*eps)/alpha.sqrt()).clamp(-1, 1)
            previous = alphas[times[position+1]] if position+1 < len(times) else torch.ones((), device=x.device)
            sample = previous.sqrt()*estimate+(1-previous).sqrt()*eps
        restored += estimate*mask
    restored = (restored+1)/2
    residual = (images-restored).abs()
    return restored, residual
