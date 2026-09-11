import math
import torch
from torch import nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file
from src.utils.config import ROOT


class FrozenDINO(nn.Module):
    def __init__(self, name: str, weights: str | None = None):
        super().__init__()
        if "dino" not in name.lower():
            raise ValueError("A DINO backbone is required; random/other replacements are not allowed")
        self.name = name
        self.model = timm.create_model(name, pretrained=weights is None, num_classes=0)
        if weights:
            self.model.load_state_dict(load_file(str(ROOT / weights)), strict=True)
        self.model.eval().requires_grad_(False)
        cfg = self.model.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485,0.456,0.406))).view(1,3,1,1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229,0.224,0.225))).view(1,3,1,1))
        self.size = cfg.get("input_size", (3,224,224))[-1]

    @torch.no_grad()
    def forward(self, images):
        self.model.eval()
        rgb = F.interpolate(images.repeat(1,3,1,1), (self.size,self.size), mode="bicubic", align_corners=False)
        tokens = self.model.forward_features((rgb-self.mean)/self.std)
        tokens = tokens[:, self.model.num_prefix_tokens:]
        side = math.isqrt(tokens.shape[1])
        if side*side != tokens.shape[1]:
            raise ValueError("Expected a square dense patch grid")
        return F.normalize(tokens.float(), dim=-1), side


def balanced_memory(tokens: torch.Tensor, maximum: int, seed: int = 42) -> torch.Tensor:
    if tokens.ndim != 3 or maximum < 5:
        raise ValueError("Memory needs B x patches x features and capacity >= k=5")
    generator = torch.Generator().manual_seed(seed)
    # Round-robin by base image gives each original equal opportunity under the cap.
    permutations = [torch.randperm(tokens.shape[1], generator=generator) for _ in range(len(tokens))]
    rows = []
    for j in range(tokens.shape[1]):
        for i in range(len(tokens)):
            if len(rows) < maximum:
                rows.append(tokens[i, permutations[i][j]])
        if len(rows) >= maximum:
            break
    return F.normalize(torch.stack(rows[:maximum]).float(), dim=-1)


def cosine_knn(query: torch.Tensor, memory: torch.Tensor, k: int = 5,
               query_chunk: int = 256, bank_chunk: int = 4096) -> torch.Tensor:
    flat = F.normalize(query.reshape(-1, query.shape[-1]), dim=-1)
    if len(memory) < k:
        raise ValueError("Normal memory is smaller than k")
    values = []
    for q in flat.split(query_chunk):
        best = torch.full((len(q), k), -float("inf"), device=q.device)
        for bank in memory.split(bank_chunk):
            similarity = q @ bank.to(q.device).T
            best = torch.cat([best, similarity], dim=1).topk(k, dim=1).values
        values.append((1-best.clamp(-1,1)).mean(dim=1))
    return torch.cat(values).reshape(query.shape[:-1])


def semantic_map(tokens, side: int, memory, size: int):
    distances = cosine_knn(tokens, memory)
    heatmap = F.interpolate(distances.reshape(-1,1,side,side), (size,size), mode="bilinear", align_corners=False)
    score = distances.topk(max(1, math.ceil(distances.shape[1]*0.05)), dim=1).values.mean(1)
    return score, heatmap
