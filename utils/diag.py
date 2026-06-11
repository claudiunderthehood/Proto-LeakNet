from __future__ import annotations

import math
from typing import Tuple

import torch
from sklearn.metrics import roc_auc_score


def sign_label_check(
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> Tuple[torch.Tensor, float, float, bool]:
    """Check if scores need sign flip based on ROC orientation."""
    y = labels.detach().cpu().numpy()
    s = scores.detach().cpu().numpy()
    try:
        auc_correct = float(roc_auc_score(y, s))
    except ValueError:
        auc_correct = math.nan
    try:
        auc_inverted = float(roc_auc_score(y, -s))
    except ValueError:
        auc_inverted = math.nan

    if math.isnan(auc_correct) or math.isnan(auc_inverted):
        return scores, auc_correct, auc_inverted, False

    flip = (auc_inverted - auc_correct) > 0.05
    oriented = -scores if flip else scores
    return oriented, auc_correct, auc_inverted, flip


def drift_report(
    ref_mean: torch.Tensor,
    ref_std: torch.Tensor,
    cur_mean: torch.Tensor,
    cur_std: torch.Tensor,
    ref_embed_mean: torch.Tensor,
    ref_embed_std: torch.Tensor,
    cur_embed_mean: torch.Tensor,
    cur_embed_std: torch.Tensor,
    *,
    split: str,
) -> str:
    """Summarise drift between reference and current golden batches."""
    delta_mu = torch.abs(cur_mean - ref_mean).max().item()
    delta_sigma = torch.abs(cur_std - ref_std).max().item()
    delta_mu_embed = torch.abs(cur_embed_mean - ref_embed_mean).max().item()
    delta_sigma_embed = torch.abs(cur_embed_std - ref_embed_std).max().item()

    warn = (
        delta_mu > 0.01
        or delta_sigma > 0.01
        or delta_mu_embed > 0.05
        or delta_sigma_embed > 0.05
    )
    status = "WARN" if warn else "OK"
    return (
        f"{split}: {status} Δμ={delta_mu:.4f} Δσ={delta_sigma:.4f} "
        f"Δμ_embed={delta_mu_embed:.4f} Δσ_embed={delta_sigma_embed:.4f}"
    )

