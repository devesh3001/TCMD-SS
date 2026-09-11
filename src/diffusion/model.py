import math
import torch
from torch import nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, inputs: int, outputs: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, inputs), inputs)
        self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, outputs), outputs)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.time = nn.Linear(time_dim, outputs)
        self.skip = nn.Conv2d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x, time):
        h = self.conv1(F.silu(self.norm1(x))) + self.time(time)[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class MaskedUNet(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()
        self.base = base
        dim = base * 4
        self.time = nn.Sequential(nn.Linear(base, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.input = nn.Conv2d(3, base, 3, padding=1)
        self.e1, self.e2 = Residual(base, base, dim), Residual(base, base*2, dim)
        self.mid = Residual(base*2, base*4, dim)
        self.d2, self.d1 = Residual(base*6, base*2, dim), Residual(base*3, base, dim)
        self.output = nn.Sequential(nn.GroupNorm(min(8, base), base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, noisy, visible, mask, timestep):
        frequency = torch.exp(-math.log(10000) * torch.arange(self.base//2, device=noisy.device) / max(1, self.base//2-1))
        phases = timestep[:, None].float() * frequency[None]
        embedding = self.time(torch.cat([phases.sin(), phases.cos()], dim=1))
        # The hidden query is never passed through the conditioning channel.
        x = self.input(torch.cat([noisy*mask + visible, visible, mask], dim=1))
        a = self.e1(x, embedding)
        b = self.e2(F.avg_pool2d(a, 2), embedding)
        c = self.mid(F.avg_pool2d(b, 2), embedding)
        d = self.d2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode="nearest"), b], dim=1), embedding)
        return self.output(self.d1(torch.cat([F.interpolate(d, size=a.shape[-2:], mode="nearest"), a], dim=1), embedding))
