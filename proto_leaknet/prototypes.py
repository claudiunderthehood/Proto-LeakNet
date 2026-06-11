from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


class ProtoHead(nn.Module):
    """Multi-prototype per class with weighted distances.

    Distances: d_{k,m} = alpha * sum_i w_i * (h_i - p_{k,m,i})^2
    Class score: D_k(h) = min_m d_{k,m} or -logsumexp(-d)
    Logits: ell_k(h) = -D_k(h)
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        protos_per_class: int = 2,
        alpha: float = 1.0,
        agg: Literal["min", "lse"] = "min",
    ):
        super().__init__()
        self.K = num_classes
        self.M = protos_per_class
        self.D = embed_dim
        self.alpha = alpha
        self.agg = agg
        self.protos = nn.Parameter(torch.randn(self.K, self.M, self.D) * 0.01)

    def distances(self, h: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
        B = h.shape[0]
        h2 = h.unsqueeze(1).unsqueeze(1).expand(B, self.K, self.M, self.D)
        p2 = self.protos.unsqueeze(0).expand(B, self.K, self.M, self.D)
        diff2 = (h2 - p2) ** 2
        if w is not None:
            w2 = w.unsqueeze(1).unsqueeze(1).expand_as(diff2)
            diff2 = diff2 * w2
        d = self.alpha * diff2.sum(dim=-1)  # [B, K, M]
        return d

    def class_scores(self, h: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
        d = self.distances(h, w)  # [B, K, M]
        if self.agg == "min":
            Dk = d.min(dim=2).values
        else:
            Dk = -torch.logsumexp(-d, dim=2)
        return Dk

    def logits(self, h: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
        return -self.class_scores(h, w)

