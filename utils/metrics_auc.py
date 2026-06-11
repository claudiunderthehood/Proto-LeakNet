from __future__ import annotations

import math
from typing import Dict, Tuple, List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def macro_auc_ovr(
    scores: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    *,
    return_per_class: bool = False,
) -> Tuple[float, Dict[int, float]] | float:
    """Macro one-vs-rest AUC across classes."""
    s = scores.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    per_class: Dict[int, float] = {}
    aucs = []
    for k in range(num_classes):
        binary = (y == k).astype(np.int32)
        if binary.sum() == 0 or binary.sum() == len(binary):
            per_class[k] = math.nan
            continue
        try:
            auc = roc_auc_score(binary, s[:, k])
        except ValueError:
            auc = math.nan
        per_class[k] = float(auc)
        if not math.isnan(auc):
            aucs.append(auc)
    macro = float(np.mean(aucs)) if aucs else math.nan
    if return_per_class:
        return macro, per_class
    return macro


def auroc(scores_closed: torch.Tensor, scores_open: torch.Tensor) -> float:
    """AUROC for closed (positive) vs open (negative) detection."""
    if scores_closed.numel() == 0 or scores_open.numel() == 0:
        return math.nan
    s_closed = scores_closed.detach().cpu().numpy()
    s_open = scores_open.detach().cpu().numpy()
    scores = np.concatenate([s_closed, s_open], axis=0)
    labels = np.concatenate([np.ones_like(s_closed), np.zeros_like(s_open)], axis=0)
    return float(roc_auc_score(labels, scores))


def ccr_crr_at_tau(
    scores_closed: torch.Tensor,
    labels_closed: torch.Tensor,
    scores_open: torch.Tensor,
    tau: float,
) -> Tuple[float, float]:
    """Compute CCR/CRR at threshold tau."""
    if scores_closed.numel() == 0 or scores_open.numel() == 0:
        return math.nan, math.nan
    s_closed = scores_closed.max(dim=1).values
    preds = scores_closed.argmax(dim=1)
    correct = preds.eq(labels_closed)
    accepted = s_closed >= tau
    ccr = (correct & accepted).float().mean().item()
    crr = (scores_open < tau).float().mean().item()
    return float(ccr), float(crr)


def oscr(
    scores_closed: torch.Tensor,
    labels_closed: torch.Tensor,
    scores_open: torch.Tensor,
) -> float:
    """Open Set Classification Rate (OSCR)."""
    if scores_closed.numel() == 0 or scores_open.numel() == 0:
        return math.nan
    conf_closed, preds = scores_closed.max(dim=1)
    correct = preds.eq(labels_closed).float()
    conf_open = scores_open.max(dim=1).values

    thresholds = torch.cat([conf_closed, conf_open]).unique()
    thresholds, _ = torch.sort(thresholds, descending=True)

    n_closed = max(float(len(conf_closed)), 1.0)
    n_open = max(float(len(conf_open)), 1.0)

    ccr_vals = []
    fpr_vals = []
    for t in thresholds:
        mask_closed = conf_closed >= t
        mask_open = conf_open >= t
        ccr_t = (correct[mask_closed].sum() / n_closed) if mask_closed.any() else 0.0
        fpr_t = (mask_open.float().sum() / n_open) if mask_open.any() else 0.0
        ccr_vals.append(float(ccr_t))
        fpr_vals.append(float(fpr_t))

    # Ensure curve ends at (0,0)
    ccr_vals.append(0.0)
    fpr_vals.append(0.0)

    order = np.argsort(fpr_vals)
    fpr_sorted = np.array(fpr_vals)[order]
    ccr_sorted = np.array(ccr_vals)[order]
    area = np.trapz(ccr_sorted, fpr_sorted)
    return float(area)


def compute_eer(scores_closed: torch.Tensor, scores_open: torch.Tensor) -> Tuple[float, float]:
    if scores_closed.numel() == 0 or scores_open.numel() == 0:
        return math.nan, math.nan
    s_closed = scores_closed.detach().cpu().numpy()
    s_open = scores_open.detach().cpu().numpy()
    scores = np.concatenate([s_closed, s_open], axis=0)
    labels = np.concatenate([np.ones_like(s_closed), np.zeros_like(s_open)], axis=0)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    idx = np.nanargmin(diff)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return float(eer), float(threshold)


def tpr_at_fpr(
    scores_closed: torch.Tensor,
    scores_open: torch.Tensor,
    *,
    target_fpr: float = 0.05,
) -> float:
    if scores_closed.numel() == 0 or scores_open.numel() == 0:
        return math.nan
    s_closed = scores_closed.detach().cpu().numpy()
    s_open = scores_open.detach().cpu().numpy()
    scores = np.concatenate([s_closed, s_open], axis=0)
    labels = np.concatenate([np.ones_like(s_closed), np.zeros_like(s_open)], axis=0)
    fpr, tpr, _ = roc_curve(labels, scores)
    target = float(target_fpr)
    if target <= fpr[0]:
        return float(tpr[0])
    if target >= fpr[-1]:
        return float(tpr[-1])
    return float(np.interp(target, fpr, tpr))


def binary_ccr_crr_at_tau(
    conf_closed: torch.Tensor,
    correct_closed: torch.Tensor,
    conf_open: torch.Tensor,
    tau: float,
) -> Tuple[float, float]:
    if conf_closed.numel() == 0 or conf_open.numel() == 0:
        return math.nan, math.nan
    accepted_closed = conf_closed >= tau
    correct = correct_closed.to(torch.float32)
    accepted_correct = correct * accepted_closed.to(torch.float32)
    ccr = accepted_correct.sum().item() / max(float(len(conf_closed)), 1.0)
    crr = (conf_open < tau).to(torch.float32).mean().item()
    return float(ccr), float(crr)


def binary_oscr(
    conf_closed: torch.Tensor,
    correct_closed: torch.Tensor,
    conf_open: torch.Tensor,
) -> float:
    if conf_closed.numel() == 0 or conf_open.numel() == 0:
        return math.nan
    correct = correct_closed.to(torch.float32)
    thresholds = torch.cat([conf_closed, conf_open]).unique()
    thresholds, _ = torch.sort(thresholds, descending=True)
    n_closed = max(float(len(conf_closed)), 1.0)
    n_open = max(float(len(conf_open)), 1.0)
    ccr_vals: List[float] = []
    fpr_vals: List[float] = []
    for t in thresholds:
        mask_closed = conf_closed >= t
        mask_open = conf_open >= t
        if mask_closed.any():
            ccr_t = (correct[mask_closed]).sum().item() / n_closed
        else:
            ccr_t = 0.0
        fpr_t = (mask_open.to(torch.float32).sum().item() / n_open) if mask_open.any() else 0.0
        ccr_vals.append(float(ccr_t))
        fpr_vals.append(float(fpr_t))
    ccr_vals.append(0.0)
    fpr_vals.append(0.0)
    order = np.argsort(fpr_vals)
    fpr_sorted = np.array(fpr_vals)[order]
    ccr_sorted = np.array(ccr_vals)[order]
    return float(np.trapz(ccr_sorted, fpr_sorted))
