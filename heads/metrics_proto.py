from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

try:
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover - sklearn optional
    GaussianMixture = None


def fit_maha_diag(
    Z_train: Tensor,
    y_train: Tensor,
    num_classes: int,
    *,
    var_floor: float = 1e-3,
    shrink: float = 0.05,
) -> Dict[str, Tensor]:
    """Fit class-wise diagonal Mahalanobis model with global z-scoring."""
    Z = torch.as_tensor(Z_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.long)
    if Z.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got shape {tuple(Z.shape)}")
    if Z.shape[0] != y.shape[0]:
        raise ValueError("Feature and label count mismatch.")

    mu = Z.mean(dim=0)
    std = Z.std(dim=0, unbiased=False).clamp_min(1e-6)
    Z_norm = (Z - mu) / std

    means = []
    inv_vars = []
    median_pool: List[Tensor] = []
    for k in range(num_classes):
        idx = (y == k).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            means.append(torch.zeros(Z.shape[1], dtype=torch.float32))
            inv_vars.append(torch.full((Z.shape[1],), 1.0 / max(var_floor, 1e-6)))
            continue
        Zk = Z_norm.index_select(0, idx)
        mean_k = Zk.mean(dim=0)
        var_k = Zk.var(dim=0, unbiased=False).clamp_min(var_floor)
        median_pool.append(var_k)
        means.append(mean_k)
        inv_vars.append(var_k)

    if median_pool:
        median_var = torch.stack(median_pool, dim=0).median(dim=0).values
    else:
        median_var = torch.ones(Z.shape[1], dtype=torch.float32)

    means_tensor = torch.stack(means, dim=0)
    inv_vars_tensor = torch.stack(inv_vars, dim=0)
    inv_vars_tensor = (1.0 - shrink) * inv_vars_tensor + shrink * median_var
    inv_vars_tensor = inv_vars_tensor.clamp_min(var_floor)
    inv_vars_tensor = 1.0 / inv_vars_tensor

    return {
        "mu": mu,
        "std": std,
        "means": means_tensor,
        "inv_vars": inv_vars_tensor,
    }


def maha_distance(
    Z: Tensor,
    model: Dict[str, Tensor],
    *,
    attention: Optional[Tensor] = None,
) -> Tensor:
    """Compute diagonal Mahalanobis distances."""
    h = torch.as_tensor(Z, dtype=torch.float32)
    mu = model["mu"].to(h.device)
    std = model["std"].to(h.device).clamp_min(1e-6)
    means = model["means"].to(h.device)
    inv_vars = model["inv_vars"].to(h.device)

    h_norm = (h - mu) / std
    diff = h_norm.unsqueeze(1) - means.unsqueeze(0)
    dist_sq = diff.pow(2)
    if attention is not None:
        att = attention.to(h.device)
        dist_sq = dist_sq * att.unsqueeze(1).clamp_min(0.0)
    dists = (dist_sq * inv_vars.unsqueeze(0)).sum(dim=-1)
    return dists


def maha_scores(
    Z: Tensor,
    model: Dict[str, Tensor],
    *,
    attention: Optional[Tensor] = None,
) -> Tensor:
    """Per-class scores = -Mahalanobis distance."""
    return -maha_distance(Z, model, attention=attention)


@dataclass
class ClassGMM:
    weights: Tensor  # [J]
    means: Tensor  # [J, D]
    vars: Tensor  # [J, D]
    valid: Tensor  # [J] boolean mask


def fit_gmm_per_class(
    Z_train: Tensor,
    y_train: Tensor,
    num_classes: int,
    *,
    J_grid: Sequence[int] = (2, 3, 4, 6),
    var_floor: float = 1e-3,
    shrink: float = 0.05,
    random_state: int = 0,
) -> Dict[str, object]:
    """Fit class-conditional diagonal GMMs for multiple component counts."""
    if GaussianMixture is None:
        raise ImportError("scikit-learn is required for Proto-GMM support.")
    Z = torch.as_tensor(Z_train, dtype=torch.float32)
    y = torch.as_tensor(y_train, dtype=torch.long)
    mu = Z.mean(dim=0)
    std = Z.std(dim=0, unbiased=False).clamp_min(1e-6)
    Z_norm = (Z - mu) / std

    models: Dict[int, Dict[str, Tensor]] = {}
    for J in J_grid:
        weights_all, means_all, vars_all, valid_all = [], [], [], []
        for k in range(num_classes):
            mask = (y == k)
            Zk = Z_norm[mask]
            class_model = _fit_diag_gmm_class(
                Zk,
                feature_dim=Z.shape[1],
                num_components=int(J),
                var_floor=var_floor,
                shrink=shrink,
                random_state=random_state + k,
            )
            weights_all.append(class_model.weights)
            means_all.append(class_model.means)
            vars_all.append(class_model.vars)
            valid_all.append(class_model.valid)
        models[int(J)] = {
            "weights": torch.stack(weights_all, dim=0),
            "means": torch.stack(means_all, dim=0),
            "vars": torch.stack(vars_all, dim=0),
            "valid": torch.stack(valid_all, dim=0),
        }

    return {
        "mu": mu,
        "std": std,
        "J_grid": [int(j) for j in J_grid],
        "models": models,
    }


def score_gmm(
    Z: Tensor,
    model: Dict[str, object],
    *,
    J: Optional[int] = None,
    return_responsibilities: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Compute per-class log-likelihoods and winning component indices."""
    h = torch.as_tensor(Z, dtype=torch.float32)
    mu = model["mu"].to(h.device)
    std = model["std"].to(h.device).clamp_min(1e-6)
    J_list: List[int] = model["J_grid"]  # type: ignore[assignment]
    chosen_J = int(J) if J is not None else int(model.get("best_J", J_list[0]))  # type: ignore[arg-type]
    if chosen_J not in model["models"]:
        raise ValueError(f"GMM with J={chosen_J} not fitted. Available: {sorted(model['models'].keys())}")
    params = model["models"][chosen_J]

    h_norm = (h - mu) / std
    weights = params["weights"].to(h.device)
    means = params["means"].to(h.device)
    vars_ = params["vars"].to(h.device).clamp_min(1e-6)
    valid = params["valid"].to(h.device)

    logps, winners, resp_store = [], [], []
    for k in range(means.shape[0]):
        log_prob, resp = _log_prob_diag_gmm(h_norm, weights[k], means[k], vars_[k], valid[k])
        logps.append(log_prob.unsqueeze(1))
        winners.append(resp.argmax(dim=1))
        if return_responsibilities:
            resp_store.append(resp)
    logps_tensor = torch.cat(logps, dim=1)
    winners_tensor = torch.stack(winners, dim=1)

    if return_responsibilities:
        resp_tensor = torch.stack(resp_store, dim=1)
        return logps_tensor, winners_tensor, resp_tensor
    return logps_tensor, winners_tensor, None


class LSEAggregator(torch.nn.Module):
    """Soft-min aggregation over prototype distances with learnable temperature."""

    def __init__(self, init_tau: float = 1.0, min_tau: float = 0.5, max_tau: float = 5.0):
        super().__init__()
        tau = torch.tensor(float(init_tau)).clamp(min=min_tau, max=max_tau)
        self.log_tau = torch.nn.Parameter(tau.log())
        self.min_tau = float(min_tau)
        self.max_tau = float(max_tau)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # tau is created from a Parameter, but be defensive:
        tau = torch.clamp(self.log_tau.exp(), self.min_tau, self.max_tau)
        # 🔧 match device & dtype (important if distances is on cuda or fp16)
        tau = tau.to(device=distances.device, dtype=distances.dtype)
        scaled = -distances / tau
        return tau * torch.logsumexp(scaled, dim=-1)

    def temperature(self) -> torch.Tensor:
        tau = torch.clamp(self.log_tau.exp(), self.min_tau, self.max_tau)
        return tau


def _fit_diag_gmm_class(
    Zk: Tensor,
    *,
    feature_dim: int,
    num_components: int,
    var_floor: float,
    shrink: float,
    random_state: int,
) -> ClassGMM:
    D = int(feature_dim)
    n = Zk.shape[0]
    if D == 0:
        weights = torch.full((num_components,), 1.0 / max(num_components, 1), dtype=torch.float32)
        means = torch.zeros((num_components, 1), dtype=torch.float32)
        vars_ = torch.ones((num_components, 1), dtype=torch.float32)
        valid = torch.zeros(num_components, dtype=torch.bool)
        if num_components:
            valid[0] = True
        return ClassGMM(weights=weights, means=means, vars=vars_, valid=valid)

    if n == 0:
        weights = torch.full((num_components,), 1.0 / max(num_components, 1), dtype=torch.float32)
        means = torch.zeros((num_components, D), dtype=torch.float32)
        vars_ = torch.ones((num_components, D), dtype=torch.float32)
        valid = torch.zeros(num_components, dtype=torch.bool)
        if num_components:
            valid[0] = True
        return ClassGMM(weights=weights, means=means, vars=vars_, valid=valid)

    n_components = max(1, min(num_components, n))
    if GaussianMixture is None:
        raise ImportError("scikit-learn is required for Proto-GMM support.")

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=var_floor,
        max_iter=256,
        random_state=random_state,
        init_params="kmeans",
    )
    gmm.fit(Zk.cpu().numpy())

    weights = torch.from_numpy(gmm.weights_).to(torch.float32)
    means = torch.from_numpy(gmm.means_).to(torch.float32)
    vars_ = torch.from_numpy(gmm.covariances_).to(torch.float32)

    median_var = vars_.median(dim=0).values
    vars_ = (1.0 - shrink) * vars_ + shrink * median_var
    vars_ = vars_.clamp_min(var_floor)

    if n_components < num_components:
        pad = num_components - n_components
        pad_weights = torch.full((pad,), 1e-6, dtype=torch.float32)
        pad_means = means[:1].repeat(pad, 1) if n_components > 0 else torch.zeros((pad, D), dtype=torch.float32)
        pad_vars = vars_[:1].repeat(pad, 1) if n_components > 0 else torch.ones((pad, D), dtype=torch.float32)
        weights = torch.cat([weights, pad_weights], dim=0)
        means = torch.cat([means, pad_means], dim=0)
        vars_ = torch.cat([vars_, pad_vars], dim=0)
        valid = torch.cat([torch.ones(n_components, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)], dim=0)
    else:
        valid = torch.ones(num_components, dtype=torch.bool)

    weights = weights / weights.sum().clamp_min(1e-12)
    return ClassGMM(weights=weights, means=means, vars=vars_, valid=valid)


def _log_prob_diag_gmm(
    Z: Tensor,
    weights: Tensor,
    means: Tensor,
    vars_: Tensor,
    valid: Tensor,
) -> Tuple[Tensor, Tensor]:
    log_weights = torch.log(weights.clamp_min(1e-12))
    diff = Z.unsqueeze(1) - means.unsqueeze(0)
    precisions = 1.0 / vars_.clamp_min(1e-6)
    log_det = torch.log(vars_.clamp_min(1e-6)).sum(dim=1)
    sq_maha = (diff.pow(2) * precisions.unsqueeze(0)).sum(dim=-1)
    const = float(Z.shape[1]) * math.log(2.0 * math.pi)
    log_comp = -0.5 * (sq_maha + log_det + const)
    mask = valid.unsqueeze(0)
    log_comp = log_comp.masked_fill(~mask, -1e12)
    log_mix = log_weights.unsqueeze(0) + log_comp
    log_prob = torch.logsumexp(log_mix, dim=1)
    resp = torch.softmax(log_mix, dim=1)
    resp = resp * mask.float()
    resp = resp / resp.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return log_prob, resp
