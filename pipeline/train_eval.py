from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import env as _env

    TELEGRAM_CHAT_ID = getattr(_env, "CHAT_ID", None)
    TOKEN_TELEGRAM = getattr(_env, "TOKEN_TELEGRAM", None)
    HF_TOKEN = getattr(_env, "TOKEN", None)
except ImportError:
    TELEGRAM_CHAT_ID = None
    TOKEN_TELEGRAM = None
    HF_TOKEN = None
from data.closed_open_dataset import DatasetBundle, build_closed_open_datasets
from heads.metrics_proto import (
    LSEAggregator,
    fit_gmm_per_class,
    fit_maha_diag,
    maha_scores,
    score_gmm,
)
from proto_leaknet.model import LeakBackbone
from proto_leaknet.prototypes import ProtoHead
from sd21.noising import (
    forward_noise,
    forward_noise_latent,
    get_noise_schedule,
    normalize_by_sigma,
    normalize_latent_by_sigma,
)
from sd21.vae import encode_latents, load_sd21_vae
from utils.diag import drift_report
from utils.metrics_auc import (
    macro_auc_ovr,
)
from sklearn.decomposition import PCA
from sklearn.metrics import auc, confusion_matrix, roc_curve, silhouette_samples
from sklearn.neighbors import KernelDensity
from proto_leaknet.compressors.dvae import LeakDVAE, DVAETrainer


class PCACompressor:
    def __init__(self, dim: int, whiten: bool = False):
        self.dim = int(dim)
        self.whiten = bool(whiten)
        self.model: Optional[PCA] = None

    def fit(self, X: np.ndarray) -> None:
        n_components = min(self.dim, X.shape[1], max(1, X.shape[0] - 1))
        self.model = PCA(n_components=n_components, whiten=self.whiten, svd_solver="randomized")
        self.model.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCACompressor must be fit before transform().")
        return self.model.transform(X).astype(np.float32, copy=False)


class DVAECompressor:
    def __init__(self, latent_dim: int, device: torch.device, epochs: int, batch_size: int, lr: float):
        self.latent_dim = int(latent_dim)
        self.device = device
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.model: Optional[LeakDVAE] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        in_dim = X.shape[1]
        dvae = LeakDVAE(in_dim=in_dim, z_dim=self.latent_dim, use_t=False)
        if getattr(dvae, "enc", None) is None:
            if getattr(dvae, "in_dim", None) is None:
                dvae.in_dim = in_dim
            dvae._init_modules()
        dvae.to(self.device)
        trainer = DVAETrainer(dvae, device=str(self.device))
        trainer.fit(
            X_train=X,
            t_train=None,
            y_train=y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=max(3, self.epochs // 4),
            outdir=None,
        )
        self.model = trainer.model.eval()

    @torch.no_grad()
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DVAECompressor must be fit before transform().")
        self.model.eval()
        tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        _, mu, _, _ = self.model(tensor)
        return mu.detach().cpu().numpy().astype(np.float32, copy=False)


def fit_llr_tied(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    shrink: float,
) -> Dict[str, torch.Tensor]:
    X = embeddings.detach()
    y = labels.detach()
    device = X.device
    dtype = X.dtype
    D = X.shape[1]
    means = []
    cov = torch.zeros(D, D, device=device, dtype=dtype)
    total = 0
    for k in range(num_classes):
        mask = y == k
        Xk = X[mask]
        if Xk.numel() == 0:
            means.append(torch.zeros(D, device=device, dtype=dtype))
            continue
        mu_k = Xk.mean(dim=0)
        means.append(mu_k)
        diff = Xk - mu_k
        cov += diff.t().mm(diff)
        total += diff.shape[0]
    denom = max(total - num_classes, 1) if total > num_classes else max(total, 1)
    cov = cov / float(denom)
    trace = torch.trace(cov)
    identity = torch.eye(D, device=device, dtype=dtype)
    alpha = float(shrink)
    cov_shrink = (1.0 - alpha) * cov + alpha * (trace / D) * identity
    cov_shrink = cov_shrink + 1e-6 * identity
    inv_cov = torch.linalg.inv(cov_shrink)
    means_tensor = torch.stack(means, dim=0)
    prior = torch.zeros(num_classes, device=device, dtype=dtype)
    for k in range(num_classes):
        prior[k] = (y == k).sum()
    if prior.sum() > 0:
        prior = (prior / prior.sum()).clamp_min(1e-6)
    else:
        prior.fill_(1.0 / num_classes)
    bias = -0.5 * torch.sum((means_tensor @ inv_cov) * means_tensor, dim=1) + torch.log(prior)
    return {
        "means": means_tensor,
        "inv_cov": inv_cov,
        "bias": bias,
    }


def llr_scores(embeddings: torch.Tensor, model: Dict[str, torch.Tensor]) -> torch.Tensor:
    means = model["means"].to(embeddings.device)
    inv_cov = model["inv_cov"].to(embeddings.device)
    bias = model["bias"].to(embeddings.device)
    diff = embeddings.unsqueeze(1) - means.unsqueeze(0)
    tmp = torch.einsum("bkd,dd->bkd", diff, inv_cov)
    quad = (tmp * diff).sum(dim=2)
    scores = -0.5 * quad + bias.unsqueeze(0)
    return scores


class AMSoftmaxHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, margin: float, scale: float):
        super().__init__()
        self.margin = float(margin)
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = F.normalize(embeddings, dim=1)
        w = F.normalize(self.weight, dim=1)
        return torch.matmul(x, w.t())

    def logits(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        cos = self.forward(embeddings)
        if labels is None:
            return self.scale * cos
        phi = cos.clone()
        idx = torch.arange(embeddings.size(0), device=embeddings.device)
        phi[idx, labels] -= self.margin
        return self.scale * phi

def fit_kde_per_class(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    bandwidth: float,
) -> Dict[str, object]:
    models: List[Optional[KernelDensity]] = []
    X = embeddings.detach().cpu().numpy().astype(np.float64, copy=False)
    y = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    for k in range(num_classes):
        Xk = X[y == k]
        if Xk.shape[0] >= 2:
            kde = KernelDensity(bandwidth=float(bandwidth), kernel="gaussian")
            kde.fit(Xk)
            models.append(kde)
        else:
            models.append(None)
    return {"models": models, "bandwidth": float(bandwidth)}


def kde_scores(embeddings: torch.Tensor, model: Dict[str, object]) -> torch.Tensor:
    X = embeddings.detach().cpu().numpy().astype(np.float64, copy=False)
    logps = []
    for kde in model.get("models", []):
        if kde is None:
            logps.append(np.full(X.shape[0], fill_value=-1e6, dtype=np.float64))
        else:
            logps.append(kde.score_samples(X))
    if not logps:
        return torch.zeros((embeddings.shape[0], 0), device=embeddings.device)
    arr = np.stack(logps, axis=1).astype(np.float32, copy=False)
    return torch.from_numpy(arr).to(embeddings.device)


@dataclass
class PipelineArgs:
    closed_root: Path
    backbone: str
    pretrained: bool
    embed_dim: int
    protos_per_class: int
    use_attention: bool
    metric: str
    var_floor: float
    shrink: float
    gmm_components: Tuple[int, ...]
    lse_temperature_learnable: bool
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    device: str
    num_workers: int
    seed: int
    t_steps: Tuple[int, ...]
    sigma_normalize: bool
    highpass: bool
    latent_space: bool
    tau_quantile: float
    calib: bool
    tta_flip: bool
    compressor: str
    pca_dim: Optional[int]
    pca_sweep: bool
    pca_sweep_dims: Tuple[int, ...]
    pca_whiten: bool
    dvae_latent_dim: int
    dvae_epochs: int
    dvae_batch_size: int
    dvae_lr: float
    kde_bandwidth: float
    kde_bandwidths: Tuple[float, ...]
    llr_shrink: float
    calibration_mode: str
    temperature_lr: float
    temperature_steps: int
    time_dropout: float
    time_entropy_weight: float
    am_softmax: bool
    am_margin: float
    am_scale: float
    am_lambda_proto: float
    am_lambda_reg: float
    extra_proto_auc_threshold: float
    extra_proto_count: int
    patience: int
    checkpoint_name: str
    run_name: str
    output_dir: Path
    train_step: Optional[str]
    make_class_list: bool


@dataclass
class FeatureSet:
    embeddings: torch.Tensor
    labels: torch.Tensor
    paths: List[str]
    attention: Optional[torch.Tensor]
    proto_scores: Optional[torch.Tensor]
    temporal_weights: Optional[List[torch.Tensor]] = field(default_factory=list)


@dataclass
class EvaluationResults:
    closed_macro_auc: float
    per_class_auc: Dict[str, float]
    top1: float
    top5: float
    balanced_acc: float
    calibration_stats: List[Tuple[float, float]]
    drift_logs: List[str]
    metric_meta: Dict[str, object]
    temperatures: Optional[List[float]]


@dataclass
class ClassificationSample:
    path: str
    true_class: str
    predicted_class: str


class TemporalAttnPool(nn.Module):
    """Simple attention pooling across T diffusion steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        logits = self.query(x).squeeze(-1)  # [B, T]
        weights = torch.softmax(logits, dim=1)
        pooled = torch.einsum("bt,btd->bd", weights, x)
        return pooled, weights


class ProtoLeakPipeline:
    def __init__(self, args: PipelineArgs):
        self.args = args
        requested_device = (args.device or "cpu").lower()
        if "cuda" in requested_device and not torch.cuda.is_available():
            print("[WARN] CUDA unavailable, falling back to CPU.")
            notify_telegram("ProtoLeak experiment falling back to CPU (CUDA unavailable).")
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self._set_seed(args.seed)

        self.latent_mode = bool(args.latent_space)
        self.latent_vae: Optional[torch.nn.Module] = None
        self.latent_scaling: float = 1.0
        self.latent_dtype: torch.dtype = torch.float32
        self.latent_input_shape: Optional[Tuple[int, int, int]] = None
        if self.latent_mode:
            self.latent_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            try:
                self.latent_vae, self.latent_scaling = load_sd21_vae(
                    self.device, dtype=self.latent_dtype, token=HF_TOKEN
                )
            except Exception as exc:
                notify_telegram(f"ProtoLeak experiment failed loading SD21 VAE: {exc}")
                raise

        self.schedule = get_noise_schedule(model="sd21", num_steps=1000)
        self.multi_t = len(args.t_steps) > 1
        self.temporal_pool = TemporalAttnPool(args.embed_dim).to(self.device) if self.multi_t else None

        self.output_dir = self.args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / self.args.checkpoint_name
        self.best_checkpoint_saved = False
        self.compressor_type = self.args.compressor.lower()
        self.compressor_model: Optional[object] = None
        self.compressor_meta: Dict[str, object] = {}
        self.calibration_mode = self.args.calibration_mode.lower()
        self.temperature_lr = self.args.temperature_lr
        self.temperature_steps = self.args.temperature_steps
        self.temperature_params: Optional[torch.Tensor] = None
        self._collect_temporal_weights = False
        self.time_dropout = float(max(0.0, min(1.0, self.args.time_dropout)))
        self.time_entropy_weight = max(0.0, self.args.time_entropy_weight)
        self.extra_proto_threshold = float(self.args.extra_proto_auc_threshold)
        self.extra_proto_count = int(max(0, self.args.extra_proto_count))
        self.prototypes_augmented = False

        self.bundle = build_closed_open_datasets(
            args.closed_root,
            None,
            backbone=args.backbone,
            pretrained=args.pretrained,
            image_size=256,
            compute_channel_stats=(not args.pretrained and not self.latent_mode),
            latent_mode=self.latent_mode,
            seed=args.seed,
        )
        # Optionally filter splits by dataset step (e.g., step1/step2/step3) across train/val/test
        self._apply_step_filter(args.train_step)
        self.num_classes = len(self.bundle.class_to_idx)
        self.am_softmax_enabled = bool(self.args.am_softmax)
        self.am_margin = self.args.am_margin
        self.am_scale = self.args.am_scale
        self.am_lambda_proto = self.args.am_lambda_proto
        self.am_lambda_reg = self.args.am_lambda_reg
        self.am_head: Optional[AMSoftmaxHead] = None

        in_channels = 4 if self.latent_mode else 3
        self.backbone = LeakBackbone(
            in_ch=in_channels,
            embed_dim=args.embed_dim,
            use_attention=args.use_attention,
            encoder_type=args.backbone,
            resnet_pretrained=args.pretrained,
            effnet_pretrained=args.pretrained,
        ).to(self.device)
        self.proto_head = ProtoHead(
            num_classes=self.num_classes,
            embed_dim=args.embed_dim,
            protos_per_class=args.protos_per_class,
            agg="min",
        ).to(self.device)
        if self.am_softmax_enabled:
            self.am_head = AMSoftmaxHead(
                embed_dim=args.embed_dim,
                num_classes=self.num_classes,
                margin=self.am_margin,
                scale=self.am_scale,
            ).to(self.device)
        self.lse_agg = LSEAggregator().to(self.device) if args.lse_temperature_learnable else None

        parameters = list(self.backbone.parameters()) + list(self.proto_head.parameters())
        if self.lse_agg is not None:
            parameters += list(self.lse_agg.parameters())
        if self.am_softmax_enabled and self.am_head is not None:
            parameters += list(self.am_head.parameters())
        self.optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader = self._make_loader(self.bundle.train, shuffle=True)
        self.val_loader = self._make_loader(self.bundle.val, shuffle=False)
        self.test_loader = self._make_loader(self.bundle.test, shuffle=False)

        self.temporal_weights_collector: List[torch.Tensor] = []

    def _make_loader(self, dataset, shuffle: bool):
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def _apply_step_filter(self, step_sel: Optional[str]) -> None:
        if not step_sel or str(step_sel).lower() in {"all", "none", "off"}:
            return
        step = str(step_sel).lower()
        valid = {"step1", "step2", "step3"}
        if step not in valid:
            print(f"[WARN] Unknown step filter '{step_sel}'. Expected one of {sorted(valid)}. Skipping.")
            return
    
        def _step_tag(path: Path) -> Optional[str]:
            for part in reversed(path.parts[:-1]):  # skip filename
                lower = part.lower()
                if lower.startswith("step"):
                    return lower
            return None

        def has_step(p: Path) -> bool:
            return _step_tag(p) == step

        try:
            from data.closed_open_dataset import ClosedSetDataset  # local import
            # Train
            tr_old = len(self.bundle.train.samples)
            tr_filtered = [s for s in self.bundle.train.samples if has_step(s.path)]
            self.bundle.train = ClosedSetDataset(tr_filtered, self.bundle.train.transform)
            tr_new = len(self.bundle.train.samples)
            # Val
            va_old = len(self.bundle.val.samples)
            va_filtered = [s for s in self.bundle.val.samples if has_step(s.path)]
            self.bundle.val = ClosedSetDataset(va_filtered, self.bundle.val.transform)
            va_new = len(self.bundle.val.samples)
            # Test
            te_old = len(self.bundle.test.samples)
            te_filtered = [s for s in self.bundle.test.samples if has_step(s.path)]
            self.bundle.test = ClosedSetDataset(te_filtered, self.bundle.test.transform)
            te_new = len(self.bundle.test.samples)
            print(
                f"[ACCEPT] Applied step filter '{step}': train {tr_new}/{tr_old}, val {va_new}/{va_old}, test {te_new}/{te_old} kept."
            )
        except Exception as exc:
            print(f"[WARN] Failed to apply step filter '{step}': {exc}")

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Training -----------------------------------------------------------------

    def train(self) -> None:
        if self.args.epochs <= 0:
            return
        best_val_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(1, self.args.epochs + 1):
            self.backbone.train()
            self.proto_head.train()
            if self.lse_agg is not None:
                self.lse_agg.train()
            if self.am_head is not None:
                self.am_head.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for images, labels, _ in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                scores, t_weights, embeddings = self._forward_scores(images)
                proto_loss = self.criterion(scores, labels)
                total_batch_loss = proto_loss
                if self.am_softmax_enabled and self.am_head is not None:
                    am_logits = self.am_head.logits(embeddings, labels)
                    am_loss = F.cross_entropy(am_logits, labels)
                    reg_loss = embeddings.norm(dim=1).mean()
                    total_batch_loss = am_loss + self.am_lambda_proto * proto_loss + self.am_lambda_reg * reg_loss
                if self.time_entropy_weight > 0 and t_weights is not None:
                    weights = t_weights / t_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1)
                    max_entropy = math.log(weights.size(1)) if weights.size(1) > 0 else 0.0
                    time_penalty = self.time_entropy_weight * (max_entropy - entropy).mean()
                    total_batch_loss = total_batch_loss + time_penalty
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]], max_norm=5.0
                )
                self.optimizer.step()

                total_loss += total_batch_loss.item() * labels.size(0)
                preds = scores.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
            avg_loss = total_loss / max(total_samples, 1)
            acc = total_correct / max(total_samples, 1)
            val_loss = self._compute_loss(self.val_loader)
            if val_loss is None:
                val_loss = avg_loss
            else:
                val_loss = float(val_loss)
            improved = val_loss < best_val_loss - 1e-5
            if improved:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self._save_checkpoint()
            else:
                epochs_no_improve += 1
            print(
                f"[Epoch {epoch}] loss={avg_loss:.4f} acc={acc:.4f} val_loss={val_loss:.4f}"
            )
            if epochs_no_improve >= self.args.patience:
                print(f"[EarlyStopping] patience reached ({self.args.patience}); stopping.")
                break
        if self.best_checkpoint_saved:
            self._load_best_checkpoint()

    # Forward helpers ----------------------------------------------------------

    def _compute_loss(self, loader: Optional[DataLoader]) -> Optional[float]:
        if loader is None:
            return None
        self.backbone.eval()
        self.proto_head.eval()
        if self.lse_agg is not None:
            self.lse_agg.eval()
        if self.am_head is not None:
            self.am_head.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                scores, t_weights, embeddings = self._forward_scores(images)
                proto_loss = self.criterion(scores, labels)
                loss = proto_loss
                if self.am_softmax_enabled and self.am_head is not None:
                    am_logits = self.am_head.logits(embeddings, labels)
                    am_loss = F.cross_entropy(am_logits, labels)
                    reg_loss = embeddings.norm(dim=1).mean()
                    loss = am_loss + self.am_lambda_proto * proto_loss + self.am_lambda_reg * reg_loss
                if self.time_entropy_weight > 0 and t_weights is not None:
                    weights = t_weights / t_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1)
                    max_entropy = math.log(weights.size(1)) if weights.size(1) > 0 else 0.0
                    time_penalty = self.time_entropy_weight * (max_entropy - entropy).mean()
                    loss = loss + time_penalty
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
        self.backbone.train()
        self.proto_head.train()
        if self.lse_agg is not None:
            self.lse_agg.train()
        if self.am_head is not None:
            self.am_head.train()
        if total_samples == 0:
            return None
        return total_loss / total_samples

    def _save_checkpoint(self) -> None:
        state = {
            "backbone": self.backbone.state_dict(),
            "proto_head": self.proto_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.lse_agg is not None:
            state["lse_agg"] = self.lse_agg.state_dict()
        if self.am_head is not None:
            state["am_head"] = self.am_head.state_dict()
        torch.save(state, self.checkpoint_path)
        self.best_checkpoint_saved = True

    def _load_best_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        try:
            state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(self.checkpoint_path, map_location=self.device)
        self.backbone.load_state_dict(state["backbone"])
        self.proto_head.load_state_dict(state["proto_head"])
        if self.lse_agg is not None and "lse_agg" in state and state["lse_agg"] is not None:
            self.lse_agg.load_state_dict(state["lse_agg"])
        if self.am_head is not None and "am_head" in state and state["am_head"] is not None:
            self.am_head.load_state_dict(state["am_head"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

    def _forward_scores(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        h, w, t_weights, _ = self._forward_embeddings(images)
        proto_scores = self._proto_scores(h, w)
        return proto_scores, t_weights, h

    def _proto_scores(self, h: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
        device = self.proto_head.protos.device
        if h.device != device:
            h = h.to(device)
        if w is not None and w.device != device:
            w = w.to(device)
        distances = self.proto_head.distances(h, w)
        if distances.device != device:
            distances = distances.to(device)
        if self.lse_agg is not None:
            return self.lse_agg(distances)
        return -distances.min(dim=-1).values

    def _forward_embeddings(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        z = self._apply_noise(images)
        if self.multi_t:
            B, T, C, H, W = z.shape
            z_flat = z.view(B * T, C, H, W)
            h_flat, w_flat = self.backbone(z_flat)
            h_flat = h_flat.view(B, T, -1)
            if w_flat is not None:
                w_flat = w_flat.view(B, T, -1)
            if self.time_dropout > 0 and self.backbone.training:
                dropout_mask = (torch.rand(B, T, device=h_flat.device) < self.time_dropout)
                all_dropped = dropout_mask.all(dim=1)
                if all_dropped.any():
                    idx = torch.where(all_dropped)[0]
                    dropout_mask[idx, 0] = False
                h_flat = h_flat.masked_fill(dropout_mask.unsqueeze(-1), 0.0)
                if w_flat is not None:
                    w_flat = w_flat.masked_fill(dropout_mask.unsqueeze(-1), 0.0)
            pooled, weights = self.temporal_pool(h_flat)
            if w_flat is not None:
                w_pooled = torch.einsum("bt,btd->bd", weights, w_flat)
            else:
                w_pooled = None
            if self._collect_temporal_weights:
                self.temporal_weights_collector.append(weights.detach().cpu())
            return pooled, w_pooled, weights, h_flat
        else:
            h, w = self.backbone(z)
            return h, w, None, None

    def _apply_noise(self, images: torch.Tensor) -> torch.Tensor:
        if self.latent_mode:
            return self._apply_latent_noise(images)
        return self._apply_image_noise(images)

    def _apply_image_noise(self, images: torch.Tensor) -> torch.Tensor:
        # Bypass SD21: feed images directly when not in latent space
        t_list = self.args.t_steps
        if len(t_list) <= 1:
            return images
        # If multiple steps requested, replicate the raw image across T
        return images.unsqueeze(1).repeat(1, len(t_list), 1, 1, 1)

    def _encode_latents(self, images: torch.Tensor) -> torch.Tensor:
        if not self.latent_mode or self.latent_vae is None:
            raise RuntimeError("Latent VAE not initialized.")
        inputs = images.to(self.device, dtype=self.latent_dtype)
        latents = encode_latents(self.latent_vae, inputs, scaling_factor=self.latent_scaling)
        latents = latents.to(self.device, dtype=torch.float32)
        if self.latent_input_shape is None:
            self.latent_input_shape = tuple(latents.shape[1:])
        return latents

    def _apply_latent_noise(self, images: torch.Tensor) -> torch.Tensor:
        z0 = self._encode_latents(images)
        B = z0.size(0)
        t_list = self.args.t_steps
        outputs: List[torch.Tensor] = []
        for t in t_list:
            t_tensor = torch.full((B,), t, device=z0.device, dtype=torch.long)
            if self.args.highpass:
                print("[WARN] High-pass filtering is disabled in latent mode.")
            z_t = forward_noise_latent(z0, t_tensor, self.schedule)
            if self.args.sigma_normalize:
                z_t = normalize_latent_by_sigma(z_t, t_tensor, self.schedule)
            outputs.append(z_t.unsqueeze(1))
        if len(outputs) == 1:
            return outputs[0].squeeze(1)
        return torch.cat(outputs, dim=1)

    # Compression --------------------------------------------------------------

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    def _fit_compressor(self, train_feats: FeatureSet, val_feats: FeatureSet) -> None:
        self.compressor_model = None
        self.compressor_meta = {}
        comp = self.compressor_type
        if comp == "none":
            return
        X_train = self._to_numpy(train_feats.embeddings)
        y_train = train_feats.labels.detach().cpu().numpy().astype(np.int64)
        if comp == "pca":
            dim_candidates = self.args.pca_sweep_dims if self.args.pca_sweep else ()
            target_dim = self.args.pca_dim
            if self.args.pca_sweep and val_feats.embeddings.numel() > 0:
                X_val = self._to_numpy(val_feats.embeddings)
                y_val = val_feats.labels.detach().cpu().numpy().astype(np.int64)
                best_dim = self._select_pca_dim(X_train, y_train, X_val, y_val, dim_candidates)
            else:
                best_dim = target_dim if target_dim is not None else min(64, X_train.shape[1])
            best_dim = max(1, min(best_dim, X_train.shape[1]))
            compressor = PCACompressor(best_dim, whiten=self.args.pca_whiten)
            compressor.fit(X_train)
            self.compressor_model = compressor
            self.compressor_meta = {"dim": best_dim, "whiten": self.args.pca_whiten}
        elif comp == "dvae":
            dvae = DVAECompressor(
                latent_dim=self.args.dvae_latent_dim,
                device=self.device,
                epochs=self.args.dvae_epochs,
                batch_size=self.args.dvae_batch_size,
                lr=self.args.dvae_lr,
            )
            dvae.fit(X_train, y_train)
            self.compressor_model = dvae
            self.compressor_meta = {"latent_dim": self.args.dvae_latent_dim}
        else:
            raise ValueError(f"Unknown compressor '{comp}'.")

    def _apply_compressor_to_features(self, feats: FeatureSet) -> FeatureSet:
        if self.compressor_model is None:
            return feats
        X = self._to_numpy(feats.embeddings)
        X_trans = self.compressor_model.transform(X)
        emb = torch.from_numpy(X_trans)
        return FeatureSet(
            embeddings=emb,
            labels=feats.labels,
            paths=feats.paths,
            attention=None,
            proto_scores=feats.proto_scores,
            temporal_weights=feats.temporal_weights,
        )

    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.compressor_model is None:
            return tensor
        X = self._to_numpy(tensor)
        X_trans = self.compressor_model.transform(X)
        return torch.from_numpy(X_trans).to(tensor.device)

    def _select_pca_dim(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        dims: Tuple[int, ...],
    ) -> int:
        best_dim = None
        best_auc = -float("inf")
        candidates = sorted({int(d) for d in dims if int(d) > 0}) or [min(64, X_train.shape[1])]
        for d in candidates:
            n_components = max(1, min(d, X_train.shape[1], X_train.shape[0] - 1))
            compressor = PCACompressor(n_components, whiten=self.args.pca_whiten)
            compressor.fit(X_train)
            Z_tr = compressor.transform(X_train)
            Z_val = compressor.transform(X_val)
            auc = self._macro_auc_for_embeddings(Z_tr, y_train, Z_val, y_val)
            if auc > best_auc:
                best_auc = auc
                best_dim = n_components
        if best_dim is None:
            best_dim = max(1, min(self.args.pca_dim or X_train.shape[1], X_train.shape[1]))
        return best_dim

    def _macro_auc_for_embeddings(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        train_tensor = torch.from_numpy(X_train)
        val_tensor = torch.from_numpy(X_val)
        y_tr = torch.from_numpy(y_train.astype(np.int64))
        y_v = torch.from_numpy(y_val.astype(np.int64))
        model = fit_maha_diag(
            train_tensor,
            y_tr,
            self.num_classes,
            var_floor=self.args.var_floor,
            shrink=self.args.shrink,
        )
        scores = maha_scores(val_tensor, model)
        macro = macro_auc_ovr(scores, y_v, self.num_classes)
        return float(macro)

    # Feature extraction -------------------------------------------------------

    def collect_features(self, loader: DataLoader) -> FeatureSet:
        self.backbone.eval()
        self.proto_head.eval()
        if self.lse_agg is not None:
            self.lse_agg.eval()
        embeddings, labels, attention, paths, proto_scores = [], [], [], [], []
        temporal_weights: List[torch.Tensor] = []
        with torch.no_grad():
            for images, y, path_list in loader:
                images = images.to(self.device, non_blocking=True)
                h, w, weights, _ = self._forward_embeddings(images)
                scores = self._proto_scores(h, w)
                embeddings.append(h.cpu())
                labels.append(y)
                paths.extend(path_list)
                proto_scores.append(scores.cpu())
                if w is not None:
                    attention.append(w.cpu())
                if weights is not None:
                    temporal_weights.append(weights.cpu())
        return FeatureSet(
            embeddings=torch.cat(embeddings, dim=0) if embeddings else torch.zeros(0, self.args.embed_dim),
            labels=torch.cat(labels, dim=0) if labels else torch.zeros(0, dtype=torch.long),
            paths=paths,
            attention=torch.cat(attention, dim=0) if attention else None,
            proto_scores=torch.cat(proto_scores, dim=0) if proto_scores else None,
            temporal_weights=temporal_weights,
        )

    # Scoring ------------------------------------------------------------------

    def compute_metric_scores(
        self,
        features: FeatureSet,
        metric_model: Dict[str, torch.Tensor],
        *,
        attention: Optional[torch.Tensor],
        metric: str,
    ) -> torch.Tensor:
        emb = features.embeddings
        att = attention
        if metric != "euclidean" and self.compressor_model is not None:
            emb = self._compress_tensor(emb)
            att = None
        if metric == "maha_diag":
            return maha_scores(emb, metric_model, attention=att)
        if metric == "proto_gmm":
            scores, _, _ = score_gmm(emb, metric_model, J=metric_model.get("best_J"))
            return scores
        if metric == "kde":
            return kde_scores(emb, metric_model)
        if metric == "llr_tied":
            return llr_scores(emb, metric_model)
        if metric == "euclidean":
            return features.proto_scores if features.proto_scores is not None else torch.zeros(0, self.num_classes)
        raise ValueError(f"Unknown metric '{metric}'.")

    def fit_metric_model(self, train_feats: FeatureSet, val_feats: FeatureSet) -> Dict[str, object]:
        metric = self.args.metric
        train_emb = train_feats.embeddings
        val_emb = val_feats.embeddings
        if metric != "euclidean" and self.compressor_model is not None:
            train_emb = self._compress_tensor(train_emb)
            val_emb = self._compress_tensor(val_emb)
        if metric == "maha_diag":
            return fit_maha_diag(
                train_emb,
                train_feats.labels,
                self.num_classes,
                var_floor=self.args.var_floor,
                shrink=self.args.shrink,
            )
        if metric == "proto_gmm":
            model = fit_gmm_per_class(
                train_emb,
                train_feats.labels,
                self.num_classes,
                J_grid=self.args.gmm_components,
                var_floor=self.args.var_floor,
                shrink=self.args.shrink,
                random_state=self.args.seed,
            )
            best_macro = float("-inf")
            best_J = None
            for J in model["J_grid"]:
                scores_val, _, _ = score_gmm(val_emb, model, J=J)
                macro, _ = macro_auc_ovr(scores_val, val_feats.labels, self.num_classes, return_per_class=True)
                if macro > best_macro:
                    best_macro = macro
                    best_J = J
            if best_J is None:
                best_J = model["J_grid"][0]
            model["best_J"] = best_J
            model["val_macro_auc"] = best_macro
            return model
        if metric == "kde":
            bw_candidates = self.args.kde_bandwidths if self.args.kde_bandwidths else (self.args.kde_bandwidth,)
            best_bw = None
            best_auc = float("-inf")
            val_labels_np = val_feats.labels.detach().cpu()
            for bw in bw_candidates:
                model_bw = fit_kde_per_class(train_emb, train_feats.labels, self.num_classes, bandwidth=bw)
                scores_val = kde_scores(val_emb, model_bw)
                macro = macro_auc_ovr(scores_val, val_labels_np, self.num_classes)
                if macro > best_auc:
                    best_auc = macro
                    best_bw = bw
                    best_model = model_bw
            if best_bw is None:
                best_model = fit_kde_per_class(train_emb, train_feats.labels, self.num_classes, bandwidth=self.args.kde_bandwidth)
                best_bw = self.args.kde_bandwidth
            best_model["bandwidth"] = float(best_bw)
            best_model["val_macro_auc"] = float(best_auc)
            return best_model
        if metric == "llr_tied":
            return fit_llr_tied(train_emb, train_feats.labels, self.num_classes, shrink=self.args.llr_shrink)
        if metric == "euclidean":
            return {}
        raise ValueError(f"Unknown metric '{metric}'.")

    def _metric_scores_from_emb(
        self,
        embeddings: torch.Tensor,
        attention: Optional[torch.Tensor],
        metric_model: Dict[str, object],
        metric: str,
    ) -> torch.Tensor:
        emb = embeddings
        att = attention
        if metric != "euclidean" and self.compressor_model is not None:
            emb = self._compress_tensor(embeddings)
            att = None
        if metric == "maha_diag":
            return maha_scores(emb, metric_model, attention=att)
        if metric == "proto_gmm":
            scores, _, _ = score_gmm(emb, metric_model, J=metric_model.get("best_J"))
            return scores
        if metric == "kde":
            return kde_scores(emb, metric_model)
        if metric == "llr_tied":
            return llr_scores(emb, metric_model)
        if metric == "euclidean":
            distances = self.proto_head.distances(embeddings, attention)
            if self.lse_agg is not None:
                return self.lse_agg(distances)
            return -distances.min(dim=-1).values
        raise ValueError(f"Unknown metric '{metric}'.")

    def _scores_from_loader(
        self,
        loader: DataLoader,
        metric_model: Dict[str, object],
        *,
        metric: str,
        apply_tta: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], Optional[torch.Tensor]]:
        scores_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        paths: List[str] = []
        attentions: List[torch.Tensor] = []
        with torch.no_grad():
            for images, labels, path_list in loader:
                images = images.to(self.device, non_blocking=True)
                embeddings, attention, _, _ = self._forward_embeddings(images)
                scores = self._metric_scores_from_emb(embeddings, attention, metric_model, metric)
                if apply_tta:
                    flipped = torch.flip(images, dims=[3])
                    emb_flip, att_flip, _, _ = self._forward_embeddings(flipped)
                    scores_flip = self._metric_scores_from_emb(emb_flip, att_flip, metric_model, metric)
                    scores = 0.5 * (scores + scores_flip)
                    if attention is not None and att_flip is not None:
                        attention = 0.5 * (attention + att_flip)
                scores_list.append(scores.cpu())
                labels_list.append(labels)
                paths.extend(path_list)
                if attention is not None:
                    attentions.append(attention.cpu())
        if scores_list:
            scores_tensor = torch.cat(scores_list, dim=0)
            labels_tensor = torch.cat(labels_list, dim=0)
            att_tensor = torch.cat(attentions, dim=0) if attentions else None
        else:
            scores_tensor = torch.zeros(0, self.num_classes)
            labels_tensor = torch.zeros(0, dtype=torch.long)
            att_tensor = None
        return scores_tensor, labels_tensor, paths, att_tensor

    def _load_batch_from_paths(self, paths: Sequence[str], transform) -> torch.Tensor:
        images: List[torch.Tensor] = []
        for p in paths:
            with Image.open(p) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                tensor = transform(img)
            images.append(tensor)
        return torch.stack(images, dim=0) if images else torch.empty(0)

    def _fit_calibration(self, scores: torch.Tensor, labels: torch.Tensor) -> List[Tuple[float, float]]:
        stats: List[Tuple[float, float]] = []
        for k in range(self.num_classes):
            mask = labels == k
            if mask.any():
                mu = scores[mask, k].mean().item()
                sigma = scores[mask, k].std(unbiased=False).item()
                if sigma < 1e-6:
                    sigma = 1.0
            else:
                mu, sigma = 0.0, 1.0
            stats.append((mu, sigma))
        return stats

    def _apply_calibration(self, scores: torch.Tensor, stats: List[Tuple[float, float]]) -> torch.Tensor:
        calibrated = scores.clone()
        for k, (mu, sigma) in enumerate(stats):
            denom = sigma if sigma > 1e-6 else 1.0
            calibrated[:, k] = (calibrated[:, k] - mu) / denom
        return calibrated

    def _fit_per_class_temperature(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = scores.device
        log_t = torch.zeros(self.num_classes, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([log_t], lr=self.temperature_lr)
        y = labels.to(device)
        for _ in range(max(1, self.temperature_steps)):
            optimizer.zero_grad()
            temps = torch.exp(log_t).clamp(min=1e-3, max=1e3)
            scaled = scores / temps.unsqueeze(0)
            loss = F.cross_entropy(scaled, y)
            loss.backward()
            optimizer.step()
        temps = torch.exp(log_t).clamp(min=1e-3, max=1e3).detach()
        return temps

    @staticmethod
    def _apply_temperature(scores: torch.Tensor, temps: torch.Tensor) -> torch.Tensor:
        if temps is None:
            return scores
        while temps.dim() < scores.dim():
            temps = temps.unsqueeze(0)
        return scores / temps

    def _augment_prototypes_if_needed(self, train_feats: FeatureSet, val_feats: FeatureSet) -> None:
        if self.extra_proto_count <= 0 or self.prototypes_augmented:
            return
        with torch.no_grad():
            val_proto_scores = self._proto_scores(val_feats.embeddings, val_feats.attention)
            _, per_class_auc = macro_auc_ovr(val_proto_scores, val_feats.labels, self.num_classes, return_per_class=True)
        bad_classes = [k for k, auc in per_class_auc.items() if not math.isnan(auc) and auc < self.extra_proto_threshold]
        if not bad_classes:
            return
        self._add_extra_prototypes(train_feats, bad_classes)
        self.prototypes_augmented = True

    def _add_extra_prototypes(self, train_feats: FeatureSet, bad_classes: List[int]) -> None:
        extra = self.extra_proto_count
        if extra <= 0:
            return
        device = train_feats.embeddings.device
        old = self.proto_head.protos.data.to(device)
        K, M, D = old.shape
        new_M = M + extra
        new = old.new_zeros(K, new_M, D)
        new[:, :M] = old
        embeddings = train_feats.embeddings.to(device)
        labels = train_feats.labels.to(device)
        for k in range(K):
            mask = labels == k
            class_emb = embeddings[mask]
            if k in bad_classes and class_emb.numel() > 0:
                idx = torch.randint(0, class_emb.shape[0], (extra,), device=old.device)
                new[k, M:] = class_emb[idx]
            else:
                new[k, M:] = old[k, :extra]
        self.proto_head.protos = nn.Parameter(new)
        self.proto_head.M = new_M

    def _select_tau(self, s_max: torch.Tensor, quantile: float) -> float:
        if s_max.numel() == 0:
            return 0.0
        q = torch.tensor(quantile, dtype=s_max.dtype, device=s_max.device)
        q = q.clamp(0.0, 1.0)
        tau = torch.quantile(s_max, q).item()
        return float(tau)

    def _topk_accuracy(self, scores: torch.Tensor, labels: torch.Tensor, topk: Tuple[int, ...]) -> List[float]:
        if scores.numel() == 0:
            return [0.0 for _ in topk]
        maxk = max(topk)
        _, pred = scores.topk(maxk, dim=1)
        labels_exp = labels.view(-1, 1).expand_as(pred)
        correct = pred.eq(labels_exp)
        res = []
        for k in topk:
            res.append(correct[:, :k].any(dim=1).float().mean().item())
        return res

    def _balanced_accuracy(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        if scores.numel() == 0:
            return 0.0
        preds = scores.argmax(dim=1)
        recalls = []
        for k in range(self.num_classes):
            mask = labels == k
            if mask.any():
                recalls.append((preds[mask] == labels[mask]).float().mean().item())
        return float(np.mean(recalls)) if recalls else 0.0

    def _embedding_statistics(
        self,
        feats: FeatureSet,
        *,
        include_silhouette: bool,
    ) -> Optional[Dict[str, object]]:
        if feats.embeddings.numel() == 0 or feats.labels.numel() == 0:
            return None
        embeddings = feats.embeddings
        if self.compressor_model is not None and self.args.metric != "euclidean":
            embeddings = self._compress_tensor(embeddings)
        emb_np = embeddings.detach().cpu().numpy()
        labels_tensor = feats.labels.detach().cpu().to(torch.long)
        labels_np = labels_tensor.numpy()
        if emb_np.shape[0] == 0 or labels_np.size == 0:
            return None
        unique = np.unique(labels_np)
        class_names = self._class_names()
        per_class_var: Dict[str, float] = {}
        per_class_std: Dict[str, float] = {}
        centroids: List[np.ndarray] = []
        for cls_idx in unique:
            cls_mask = labels_np == cls_idx
            if not np.any(cls_mask):
                continue
            cls_emb = emb_np[cls_mask]
            cls_name = class_names[int(cls_idx)] if int(cls_idx) < len(class_names) else str(cls_idx)
            var_val = float(np.var(cls_emb, axis=0).mean())
            std_val = float(np.std(cls_emb, axis=0).mean())
            per_class_var[cls_name] = var_val
            per_class_std[cls_name] = std_val
            centroids.append(cls_emb.mean(axis=0))

        def _sanitize(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            return float(value)

        intra_vals = list(per_class_var.values())
        intra_std_vals = list(per_class_std.values())
        mean_intra_var = _sanitize(float(np.mean(intra_vals))) if intra_vals else None
        mean_intra_std = _sanitize(float(np.mean(intra_std_vals))) if intra_std_vals else None

        inter_dists: List[float] = []
        if len(centroids) >= 2:
            centroids_arr = np.stack(centroids, axis=0)
            for i in range(centroids_arr.shape[0]):
                for j in range(i + 1, centroids_arr.shape[0]):
                    inter_dists.append(float(np.linalg.norm(centroids_arr[i] - centroids_arr[j])))
        dist_mean = _sanitize(float(np.mean(inter_dists))) if inter_dists else None
        dist_min = _sanitize(float(np.min(inter_dists))) if inter_dists else None
        dist_max = _sanitize(float(np.max(inter_dists))) if inter_dists else None

        snr = None
        if dist_mean is not None and mean_intra_std is not None and mean_intra_std > 0:
            snr = _sanitize(dist_mean / max(mean_intra_std, 1e-12))

        silhouette_info: Optional[Dict[str, object]] = None
        if include_silhouette and len(unique) >= 2 and emb_np.shape[0] > len(unique):
            try:
                sil_values = silhouette_samples(emb_np, labels_np)
                sil_mean = _sanitize(float(np.mean(sil_values)))
                sil_std = _sanitize(float(np.std(sil_values)))
                paths = feats.paths if feats.paths else [str(i) for i in range(len(sil_values))]
                per_sample = {paths[i]: _sanitize(float(sil_values[i])) for i in range(len(sil_values))}
                silhouette_info = {
                    "mean": sil_mean,
                    "std": sil_std,
                    "per_sample": per_sample,
                }
            except Exception as exc:
                print(f"[WARN] Failed to compute silhouette scores: {exc}")

        return {
            "intra_class_variance": {k: _sanitize(v) for k, v in per_class_var.items()},
            "mean_intra_class_variance": mean_intra_var,
            "mean_intra_class_std": mean_intra_std,
            "inter_class_centroid_distance": {
                "mean": dist_mean,
                "min": dist_min,
                "max": dist_max,
            },
            "snr": snr,
            "silhouette": silhouette_info,
        }

    def _compute_golden_drift(self) -> List[str]:
        golden = self.bundle.golden_batch
        drift_logs: List[str] = []
        baseline: Optional[Dict[str, torch.Tensor]] = None
        for split in ["train", "val", "test"]:
            paths = golden.get(split, [])
            if not paths:
                continue
            transform = self.bundle.train.transform if split == "train" else self.bundle.val.transform
            batch = self._load_batch_from_paths(paths, transform)
            if batch.numel() == 0:
                continue
            batch = batch.to(self.device)
            with torch.no_grad():
                embeddings, _, _, _ = self._forward_embeddings(batch)
            if self.latent_mode:
                latents_z0 = self._encode_latents(batch)
                channel_tensor = latents_z0.detach().cpu()
            else:
                channel_tensor = batch.detach().cpu()
            channel_mean = channel_tensor.mean(dim=[0, 2, 3])
            channel_std = channel_tensor.std(dim=[0, 2, 3])
            embed_mean = embeddings.mean(dim=0).cpu()
            embed_std = embeddings.std(dim=0).cpu()
            if baseline is None:
                baseline = {
                    "channel_mean": channel_mean,
                    "channel_std": channel_std,
                    "embed_mean": embed_mean,
                    "embed_std": embed_std,
                }
                continue
            drift_logs.append(
                drift_report(
                    baseline["channel_mean"],
                    baseline["channel_std"],
                    channel_mean,
                    channel_std,
                    baseline["embed_mean"],
                    baseline["embed_std"],
                    embed_mean,
                    embed_std,
                    split=split,
                )
            )
        return drift_logs

    def evaluate(self) -> EvaluationResults:
        self.temporal_weights_collector.clear()
        train_feats = self.collect_features(self.train_loader)
        val_feats = self.collect_features(self.val_loader)
        self._augment_prototypes_if_needed(train_feats, val_feats)
        if self.prototypes_augmented:
            with torch.no_grad():
                train_feats.proto_scores = self._proto_scores(train_feats.embeddings, train_feats.attention)
                if val_feats.embeddings.numel():
                    val_feats.proto_scores = self._proto_scores(val_feats.embeddings, val_feats.attention)
        self._fit_compressor(train_feats, val_feats)
        self._collect_temporal_weights = True
        test_feats = self.collect_features(self.test_loader)
        self._collect_temporal_weights = False
        metric_model = self.fit_metric_model(train_feats, val_feats)
        metric_meta: Dict[str, object] = {
            "metric": self.args.metric,
            "var_floor": self.args.var_floor,
            "shrink": self.args.shrink,
        }
        if isinstance(metric_model, dict):
            if "best_J" in metric_model:
                metric_meta["best_J"] = metric_model["best_J"]
            if "val_macro_auc" in metric_model:
                metric_meta["val_macro_auc"] = metric_model["val_macro_auc"]
            if self.args.metric == "kde":
                metric_meta["bandwidth"] = metric_model.get("bandwidth", self.args.kde_bandwidth)
            if self.args.metric == "llr_tied":
                metric_meta["llr_shrink"] = self.args.llr_shrink

        val_scores_raw, val_labels, _, _ = self._scores_from_loader(
            self.val_loader, metric_model, metric=self.args.metric, apply_tta=self.args.tta_flip
        )
        test_scores_raw, test_labels, test_paths, _ = self._scores_from_loader(
            self.test_loader, metric_model, metric=self.args.metric, apply_tta=self.args.tta_flip
        )
        calibration_stats: List[Tuple[float, float]] = []
        self.temperature_params = None
        val_scores, test_scores = val_scores_raw, test_scores_raw
        if self.args.calib and self.calibration_mode == "perclass_temp":
            temps = self._fit_per_class_temperature(val_scores_raw, val_labels)
            self.temperature_params = temps.detach().cpu()
            val_scores = self._apply_temperature(val_scores_raw, temps)
            test_scores = self._apply_temperature(test_scores_raw, temps)
            calibration_stats = [(float(t), 0.0) for t in self.temperature_params.tolist()]
        elif self.args.calib and self.calibration_mode == "zscore":
            calibration_stats = self._fit_calibration(val_scores_raw, val_labels)
            val_scores = self._apply_calibration(val_scores_raw, calibration_stats)
            test_scores = self._apply_calibration(test_scores_raw, calibration_stats)
        elif not self.args.calib or self.calibration_mode == "none":
            calibration_stats = [(0.0, 1.0) for _ in range(self.num_classes)]

        train_scores_raw = self._metric_scores_from_emb(
            train_feats.embeddings, train_feats.attention, metric_model, self.args.metric
        )
        if self.args.calib and self.calibration_mode == "perclass_temp" and self.temperature_params is not None:
            train_scores = self._apply_temperature(train_scores_raw, self.temperature_params.to(train_scores_raw.device))
        elif self.args.calib and self.calibration_mode == "zscore":
            train_scores = self._apply_calibration(train_scores_raw, calibration_stats)
        else:
            train_scores = train_scores_raw

        closed_macro_auc, per_class_auc_map = macro_auc_ovr(
            test_scores, test_labels, self.num_classes, return_per_class=True
        )
        top1, top5 = self._topk_accuracy(test_scores, test_labels, topk=(1, 5))
        balanced = self._balanced_accuracy(test_scores, test_labels)

        preds_test = test_scores.argmax(dim=1)
        correct_closed = preds_test.eq(test_labels)
        classification_samples: Optional[Dict[str, List[ClassificationSample]]] = None
        if self.args.make_class_list:
            classification_samples = self._collect_classification_samples(test_paths, test_labels, preds_test)

        drift_logs = self._compute_golden_drift()
        temporal_summary = self._temporal_summary()

        class_names = self._class_names()
        per_class_named = {class_names[k]: float(per_class_auc_map.get(k, float("nan"))) for k in range(self.num_classes)}

        results = EvaluationResults(
            closed_macro_auc=float(closed_macro_auc),
            per_class_auc=per_class_named,
            top1=float(top1),
            top5=float(top5),
            balanced_acc=float(balanced),
            calibration_stats=calibration_stats,
            drift_logs=drift_logs,
            metric_meta=metric_meta,
            temperatures=self.temperature_params.tolist() if self.temperature_params is not None else None,
        )

        train_shuffle_metrics: Optional[Dict[str, float]] = None
        if train_scores.numel():
            try:
                g_train = torch.Generator()
                g_train.manual_seed(self.args.seed + 223344)
                perm_train = torch.randperm(train_feats.labels.numel(), generator=g_train)
                labels_train_shuffled = train_feats.labels[perm_train]
                macro_train_shuf = macro_auc_ovr(train_scores, labels_train_shuffled, self.num_classes)
                top1_train_shuf, top5_train_shuf = self._topk_accuracy(train_scores, labels_train_shuffled, topk=(1, 5))
                balanced_train_shuf = self._balanced_accuracy(train_scores, labels_train_shuffled)
                train_shuffle_metrics = {
                    "macro_auc_closed": float(macro_train_shuf),
                    "top1": float(top1_train_shuf),
                    "top5": float(top5_train_shuf),
                    "balanced_acc": float(balanced_train_shuf),
                }
            except Exception as exc:
                print(f"[WARN] Train label-shuffle sanity failed: {exc}")

        # Label-shuffle sanity metrics to check leakage across splits (test set)
        test_shuffle_metrics: Optional[Dict[str, float]] = None
        try:
            g = torch.Generator()
            g.manual_seed(self.args.seed + 12345)
            if train_feats.labels.numel() > 0 and val_feats.labels.numel() > 0:
                perm_tr = torch.randperm(train_feats.labels.numel(), generator=g)
                perm_v = torch.randperm(val_feats.labels.numel(), generator=g)
                train_shuf = FeatureSet(
                    embeddings=train_feats.embeddings,
                    labels=train_feats.labels[perm_tr],
                    paths=train_feats.paths,
                    attention=train_feats.attention,
                    proto_scores=train_feats.proto_scores,
                    temporal_weights=train_feats.temporal_weights,
                )
                val_shuf = FeatureSet(
                    embeddings=val_feats.embeddings,
                    labels=val_feats.labels[perm_v],
                    paths=val_feats.paths,
                    attention=val_feats.attention,
                    proto_scores=val_feats.proto_scores,
                    temporal_weights=val_feats.temporal_weights,
                )
                metric_model_shuf = self.fit_metric_model(train_shuf, val_shuf)
                val_scores_shuf, val_labels_shuf, _, _ = self._scores_from_loader(
                    self.val_loader, metric_model_shuf, metric=self.args.metric, apply_tta=self.args.tta_flip
                )
                test_scores_shuf, test_labels_real, _, _ = self._scores_from_loader(
                    self.test_loader, metric_model_shuf, metric=self.args.metric, apply_tta=self.args.tta_flip
                )
                if self.args.calib and self.calibration_mode == "perclass_temp":
                    temps_shuf = self._fit_per_class_temperature(val_scores_shuf, val_labels_shuf)
                    test_scores_shuf = self._apply_temperature(test_scores_shuf, temps_shuf.to(test_scores_shuf.device))
                elif self.args.calib and self.calibration_mode == "zscore":
                    calibration_stats_shuf = self._fit_calibration(val_scores_shuf, val_labels_shuf)
                    test_scores_shuf = self._apply_calibration(test_scores_shuf, calibration_stats_shuf)
                macro_shuf = macro_auc_ovr(test_scores_shuf, test_labels_real, self.num_classes)
                top1_shuf, top5_shuf = self._topk_accuracy(test_scores_shuf, test_labels_real, topk=(1, 5))
                bal_shuf = self._balanced_accuracy(test_scores_shuf, test_labels_real)
                test_shuffle_metrics = {
                    "macro_auc_closed": float(macro_shuf),
                    "top1": float(top1_shuf),
                    "top5": float(top5_shuf),
                    "balanced_acc": float(bal_shuf),
                }
        except Exception as exc:
            print(f"[WARN] Label-shuffle sanity check failed: {exc}")

        embedding_stats: Dict[str, Optional[Dict[str, object]]] = {
            "train": self._embedding_statistics(train_feats, include_silhouette=False),
            "val": self._embedding_statistics(val_feats, include_silhouette=False),
            "test": self._embedding_statistics(test_feats, include_silhouette=True),
        }
        silhouette_path: Optional[Path] = None
        test_stats = embedding_stats.get("test")
        if test_stats is not None:
            sil_section = test_stats.get("silhouette") if isinstance(test_stats, dict) else None
            if isinstance(sil_section, dict):
                per_sample = sil_section.get("per_sample")
                if isinstance(per_sample, dict) and per_sample:
                    silhouette_path = self.metrics_dir / f"{self.args.run_name}_silhouette.json"
                    with silhouette_path.open("w", encoding="utf-8") as f:
                        json.dump(per_sample, f, indent=2)
                    sil_section["per_sample_path"] = str(silhouette_path)
                    sil_section["per_sample_count"] = len(per_sample)
                    sil_section.pop("per_sample", None)

        self._print_acceptance_logs(
            results,
            calibration_stats,
            per_class_named,
            drift_logs,
            temporal_summary,
            train_shuffle_metrics,
            test_shuffle_metrics,
            embedding_stats,
            silhouette_path,
        )
        self._dump_metrics(
            results,
            per_class_named,
            calibration_stats,
            drift_logs,
            temporal_summary,
            train_shuffle_metrics,
            test_shuffle_metrics,
            embedding_stats,
            silhouette_path,
            classification_samples,
        )

        # Confusion matrices (closed-set) — mandatory plots
        try:
            # Predictions
            def _preds(scores: torch.Tensor) -> np.ndarray:
                return scores.detach().cpu().argmax(dim=1).numpy().astype(np.int64, copy=False)

            y_tr = train_feats.labels.detach().cpu().numpy().astype(np.int64, copy=False)
            y_v = val_labels.detach().cpu().numpy().astype(np.int64, copy=False)
            y_te = test_labels.detach().cpu().numpy().astype(np.int64, copy=False)
            yhat_tr = _preds(train_scores)
            yhat_v = _preds(val_scores)
            yhat_te = _preds(test_scores)

            labels_range = list(range(self.num_classes))
            cm_tr = confusion_matrix(y_tr, yhat_tr, labels=labels_range) if y_tr.size else None
            cm_v = confusion_matrix(y_v, yhat_v, labels=labels_range) if y_v.size else None
            cm_te = confusion_matrix(y_te, yhat_te, labels=labels_range) if y_te.size else None

            plots_dir = Path("plots") / "matrixes"
            plots_dir.mkdir(parents=True, exist_ok=True)
            names = self._class_names()

            def _plot_cm(cm: np.ndarray, title: str, out_path: Path) -> None:
                if cm is None:
                    return
                K = cm.shape[0]
                fig_w = max(8, 0.5 * K)
                fig_h = max(6, 0.5 * K)
                plt.figure(figsize=(fig_w, fig_h))
                sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=names, yticklabels=names)
                plt.title(title)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(out_path, dpi=200)
                plt.close()

            # Save: main plot named as run (TEST), plus split-specific
            if cm_te is not None:
                _plot_cm(cm_te, f"{self.args.run_name} — Test (closed)", plots_dir / f"{self.args.run_name}.png")
            if cm_tr is not None:
                _plot_cm(cm_tr, f"{self.args.run_name} — Train (closed)", plots_dir / f"{self.args.run_name}_train.png")
            if cm_v is not None:
                _plot_cm(cm_v, f"{self.args.run_name} — Val (closed)", plots_dir / f"{self.args.run_name}_val.png")
            print(f"[ACCEPT] Confusion matrices saved under {plots_dir}")
        except Exception as exc:
            print(f"[WARN] Failed to create confusion matrix plots: {exc}")

        # Multiclass ROC curves (one-vs-rest) with per-class AUC in legend
        try:
            roc_dir = Path("plots") / "roc_curves"
            roc_dir.mkdir(parents=True, exist_ok=True)
            names = self._class_names()

            def _plot_roc(scores: torch.Tensor, labels: torch.Tensor, title: str, out_path: Path) -> None:
                y_true = labels.detach().cpu().numpy().astype(np.int64, copy=False)
                if y_true.size == 0:
                    return
                s_np = scores.detach().cpu().numpy().astype(np.float64, copy=False)
                plt.figure(figsize=(10, 8))
                plotted = 0
                for k in range(self.num_classes):
                    y_bin = (y_true == k).astype(np.int32, copy=False)
                    if y_bin.min() == y_bin.max():
                        continue
                    fpr, tpr, _ = roc_curve(y_bin, s_np[:, k])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=1.8, label=f"{names[k]} (AUC={roc_auc:.3f})")
                    plotted += 1
                plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Chance")
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.05)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(title)
                if plotted:
                    plt.legend(loc="lower right", fontsize=8)
                else:
                    plt.text(0.5, 0.5, "No valid ROC curves (missing positive/negative class samples).", ha="center", va="center")
                plt.tight_layout()
                plt.savefig(out_path, dpi=200)
                plt.close()

            _plot_roc(test_scores, test_labels, f"{self.args.run_name} — ROC Test (closed)", roc_dir / f"{self.args.run_name}.png")
            _plot_roc(train_scores, train_feats.labels, f"{self.args.run_name} — ROC Train (closed)", roc_dir / f"{self.args.run_name}_train.png")
            _plot_roc(val_scores, val_labels, f"{self.args.run_name} — ROC Val (closed)", roc_dir / f"{self.args.run_name}_val.png")
            print(f"[ACCEPT] ROC curves saved under {roc_dir}")
        except Exception as exc:
            print(f"[WARN] Failed to create ROC curve plots: {exc}")

        return results

    def _class_names(self) -> List[str]:
        return sorted(self.bundle.class_to_idx, key=self.bundle.class_to_idx.get)

    def _collect_classification_samples(
        self,
        paths: Sequence[str],
        labels: torch.Tensor,
        preds: torch.Tensor,
        *,
        max_correct: int = 5,
        max_wrong: int = 5,
    ) -> Dict[str, List[ClassificationSample]]:
        names = self._class_names()
        y_true = labels.detach().cpu().tolist()
        y_pred = preds.detach().cpu().tolist()
        correct_samples: List[ClassificationSample] = []
        wrong_samples: List[ClassificationSample] = []

        for path, true_idx, pred_idx in zip(paths, y_true, y_pred):
            if len(correct_samples) >= max_correct and len(wrong_samples) >= max_wrong:
                break
            file_name = Path(path).name
            true_class = names[int(true_idx)] if 0 <= int(true_idx) < len(names) else f"class_{int(true_idx)}"
            pred_class = names[int(pred_idx)] if 0 <= int(pred_idx) < len(names) else f"class_{int(pred_idx)}"

            sample = ClassificationSample(
                path=file_name,
                true_class=true_class,
                predicted_class=pred_class,
            )
            if int(pred_idx) == int(true_idx):
                if len(correct_samples) < max_correct:
                    correct_samples.append(sample)
            elif len(wrong_samples) < max_wrong:
                wrong_samples.append(sample)

        return {"correct": correct_samples, "misclassified": wrong_samples}

    def _temporal_summary(self) -> Optional[dict]:
        if not self.multi_t or not self.temporal_weights_collector:
            return None
        weights = torch.cat(self.temporal_weights_collector, dim=0)
        mean = weights.mean(dim=0)
        std = weights.std(dim=0)
        return {"mean": mean.tolist(), "std": std.tolist()}

    def _print_acceptance_logs(
        self,
        results: EvaluationResults,
        calibration_stats: List[Tuple[float, float]],
        per_class_auc: Dict[str, float],
        drift_logs: List[str],
        temporal_summary: Optional[dict],
        train_shuffle_metrics: Optional[Dict[str, float]] = None,
        test_shuffle_metrics: Optional[Dict[str, float]] = None,
        embedding_stats: Optional[Dict[str, Optional[Dict[str, object]]]] = None,
        silhouette_path: Optional[Path] = None,
    ) -> None:
        meta = self.bundle.transform_meta
        logs = []
        logs.append(
            "[ACCEPT] Backbone={backbone} pretrained={pretrained} transform={transform} size={size} (no double Normalize={dd}).".format(
                backbone=meta.get("backbone"),
                pretrained=meta.get("pretrained"),
                transform=meta.get("transform_name"),
                size=meta.get("image_size"),
                dd=meta.get("double_normalize", "false"),
            )
        )
        space = "SD21-latent" if self.latent_mode else "image"
        channels = 4 if self.latent_mode else 3
        shape_note = ""
        if self.latent_mode and self.latent_input_shape is not None:
            shape_note = f" {self.latent_input_shape[1]}x{self.latent_input_shape[2]}"
        scaling_note = f" scaling={self.latent_scaling:.5f}" if self.latent_mode else ""
        logs.append(f"[ACCEPT] Input space={space} channels={channels}{shape_note}{scaling_note}.")
        if self.latent_mode:
            logs.append(
                f"[ACCEPT] SD21 t={list(self.args.t_steps)} sigma_norm={'on' if self.args.sigma_normalize else 'off'} highpass={'on' if self.args.highpass else 'off'}."
            )
        else:
            logs.append("[ACCEPT] Image mode: bypass SD21 (feeding raw images).")
        if self.args.metric == "llr_tied":
            metric_line = f"[ACCEPT] Metric={self.args.metric} shrink={self.args.llr_shrink:.2f}"
        else:
            metric_line = (
                f"[ACCEPT] Metric={self.args.metric} var_floor={self.args.var_floor:.2e} shrink={self.args.shrink:.2f}"
            )
        if "best_J" in results.metric_meta:
            metric_line += f" best_J={results.metric_meta['best_J']} val_macro_auc={results.metric_meta.get('val_macro_auc', float('nan')):.4f}"
        logs.append(metric_line)
        comp_line = f"[ACCEPT] Compressor={self.compressor_type}"
        if self.compressor_type == "pca":
            comp_line += " dim={dim} whiten={whiten}".format(
                dim=self.compressor_meta.get("dim", "-"),
                whiten=self.compressor_meta.get("whiten", False),
            )
        elif self.compressor_type == "dvae":
            comp_line += " latent_dim={}".format(self.compressor_meta.get("latent_dim", "-"))
        else:
            comp_line += " (none)"
        logs.append(comp_line)
        logs.append(
            f"[ACCEPT] Temporal regularization dropout={self.time_dropout:.2f} entropy_weight={self.time_entropy_weight:.2f}."
        )
        if self.temperature_params is not None:
            temps_str = ", ".join(f"T{k}={t:.2f}" for k, t in enumerate(self.temperature_params.tolist()))
            logs.append(f"[ACCEPT] Per-class temperature scaling ({self.calibration_mode}): {temps_str}.")
        calib_parts = [
            f"{name}:μ={mu:.3f},σ={sigma:.3f}" for name, (mu, sigma) in zip(self._class_names(), calibration_stats)
        ]
        logs.append(
            f"[ACCEPT] Calibration per-class [{'; '.join(calib_parts)}]."
        )
        logs.append(
            f"[ACCEPT] Closed macroAUC={results.closed_macro_auc:.4f} top1={results.top1:.4f} top5={results.top5:.4f} balanced={results.balanced_acc:.4f}."
        )
        if self.args.train_step and str(self.args.train_step).lower() != "all":
            logs.append(f"[ACCEPT] Step filter across splits: {self.args.train_step}.")
        per_class_entries = ", ".join(f"{name}:{auc:.4f}" for name, auc in per_class_auc.items())
        logs.append(f"[ACCEPT] Per-class AUC [{per_class_entries}].")
        for drift in drift_logs:
            logs.append(f"[ACCEPT] Drift {drift}")
        if temporal_summary is not None:
            weights_info = ", ".join(
                f"t{idx}:{mean:.3f}±{std:.3f}"
                for idx, (mean, std) in enumerate(zip(temporal_summary["mean"], temporal_summary["std"]))
            )
            logs.append(f"[ACCEPT] TemporalAttnPool weights {weights_info}.")

        def _fmt(value: Optional[float]) -> str:
            if value is None:
                return "nan"
            try:
                val = float(value)
            except (TypeError, ValueError):
                return "nan"
            if math.isnan(val) or math.isinf(val):
                return "nan"
            return f"{val:.4f}"

        if embedding_stats:
            for split_name, stats in embedding_stats.items():
                if not stats:
                    continue
                inter = stats.get("inter_class_centroid_distance") if isinstance(stats, dict) else None
                inter_mean = None
                if isinstance(inter, dict):
                    inter_mean = inter.get("mean")
                intra_mean = stats.get("mean_intra_class_variance") if isinstance(stats, dict) else None
                snr_val = stats.get("snr") if isinstance(stats, dict) else None
                logs.append(
                    "[ACCEPT] {split} embedding stats intra_var_mean={intra} inter_centroid_mean={inter} SNR={snr}.".format(
                        split=split_name.capitalize(),
                        intra=_fmt(intra_mean),
                        inter=_fmt(inter_mean),
                        snr=_fmt(snr_val),
                    )
                )

        if train_shuffle_metrics is not None:
            logs.append(
                "[ACCEPT] Train label-shuffle sanity MacroAUC={macro} Top1={t1} Top5={t5} BalancedAcc={bal}.".format(
                    macro=_fmt(train_shuffle_metrics.get("macro_auc_closed")),
                    t1=_fmt(train_shuffle_metrics.get("top1")),
                    t5=_fmt(train_shuffle_metrics.get("top5")),
                    bal=_fmt(train_shuffle_metrics.get("balanced_acc")),
                )
            )
        if test_shuffle_metrics is not None:
            logs.append(
                "[ACCEPT] Test label-shuffle sanity MacroAUC={macro} Top1={t1} Top5={t5} BalancedAcc={bal}.".format(
                    macro=_fmt(test_shuffle_metrics.get("macro_auc_closed")),
                    t1=_fmt(test_shuffle_metrics.get("top1")),
                    t5=_fmt(test_shuffle_metrics.get("top5")),
                    bal=_fmt(test_shuffle_metrics.get("balanced_acc")),
                )
            )
        if silhouette_path is not None:
            logs.append(f"[ACCEPT] Test silhouette per-sample saved at {silhouette_path}.")

        for line in logs:
            print(line)

    def _dump_metrics(
        self,
        results: EvaluationResults,
        per_class_auc: Dict[str, float],
        calibration_stats: List[Tuple[float, float]],
        drift_logs: List[str],
        temporal_summary: Optional[dict],
        train_shuffle_metrics: Optional[Dict[str, float]] = None,
        test_shuffle_metrics: Optional[Dict[str, float]] = None,
        embedding_stats: Optional[Dict[str, Optional[Dict[str, object]]]] = None,
        silhouette_path: Optional[Path] = None,
        classification_samples: Optional[Dict[str, List[ClassificationSample]]] = None,
    ) -> None:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        path = self.metrics_dir / f"{self.args.run_name}.txt"

        def fmt(value: Optional[float]) -> str:
            if value is None:
                return "nan"
            try:
                val = float(value)
            except (TypeError, ValueError):
                return "nan"
            if math.isnan(val) or math.isinf(val):
                return "nan"
            return f"{val:.4f}"

        with path.open("w", encoding="utf-8") as f:
            f.write(f"Run: {self.args.run_name}\n")
            f.write(f"Closed root: {self.args.closed_root}\n")
            f.write(f"Backbone: {self.args.backbone} (pretrained={self.args.pretrained})\n")
            f.write(f"Latent space: {self.latent_mode}\n")
            if self.args.train_step and str(self.args.train_step).lower() != "all":
                f.write(f"Step filter across splits: {self.args.train_step}\n")
            f.write(
                "Compressor: {comp} meta={meta}\n".format(
                    comp=self.compressor_type,
                    meta=self.compressor_meta or {},
                )
            )
            f.write(f"Metric: {self.args.metric} (var_floor={self.args.var_floor:.2e}, shrink={self.args.shrink:.2f})\n")
            f.write(f"Temporal steps: {list(self.args.t_steps)} sigma_norm={self.args.sigma_normalize}\n")
            f.write(f"MacroAUC (closed): {results.closed_macro_auc:.4f}\n")
            f.write(f"Top1: {results.top1:.4f} Top5: {results.top5:.4f} BalancedAcc: {results.balanced_acc:.4f}\n")
            if results.temperatures is not None:
                f.write("Per-class temperatures:\n")
                for name, temp in zip(self._class_names(), results.temperatures):
                    f.write(f"  {name}: T={temp:.4f}\n")
            f.write("Per-class macro AUC:\n")
            for name, auc in per_class_auc.items():
                f.write(f"  {name}: {fmt(auc)}\n")
            f.write("Calibration stats (μ, σ):\n")
            for name, stats in zip(self._class_names(), calibration_stats):
                mu, sigma = stats
                f.write(f"  {name}: mu={mu:.4f} sigma={sigma:.4f}\n")
            if drift_logs:
                f.write("Drift logs:\n")
                for entry in drift_logs:
                    f.write(f"  {entry}\n")
            if temporal_summary is not None:
                f.write("Temporal attention weights:\n")
                for idx, (mean, std) in enumerate(zip(temporal_summary.get("mean", []), temporal_summary.get("std", []))):
                    f.write(f"  t{idx}: mean={mean:.4f} std={std:.4f}\n")
            if train_shuffle_metrics is not None or test_shuffle_metrics is not None:
                f.write("Label-shuffle sanity:\n")
                if train_shuffle_metrics is not None:
                    f.write(
                        "  Train -> MacroAUC={macro} Top1={t1} Top5={t5} BalancedAcc={bal}\n".format(
                            macro=fmt(train_shuffle_metrics.get("macro_auc_closed")),
                            t1=fmt(train_shuffle_metrics.get("top1")),
                            t5=fmt(train_shuffle_metrics.get("top5")),
                            bal=fmt(train_shuffle_metrics.get("balanced_acc")),
                        )
                    )
                if test_shuffle_metrics is not None:
                    f.write(
                        "  Test -> MacroAUC={macro} Top1={t1} Top5={t5} BalancedAcc={bal}\n".format(
                            macro=fmt(test_shuffle_metrics.get("macro_auc_closed")),
                            t1=fmt(test_shuffle_metrics.get("top1")),
                            t5=fmt(test_shuffle_metrics.get("top5")),
                            bal=fmt(test_shuffle_metrics.get("balanced_acc")),
                        )
                    )
            if embedding_stats:
                f.write("Embedding statistics:\n")
                for split, stats in embedding_stats.items():
                    if not stats:
                        continue
                    inter = stats.get("inter_class_centroid_distance") if isinstance(stats, dict) else None
                    inter_mean = inter.get("mean") if isinstance(inter, dict) else None
                    inter_min = inter.get("min") if isinstance(inter, dict) else None
                    inter_max = inter.get("max") if isinstance(inter, dict) else None
                    snr_val = stats.get("snr") if isinstance(stats, dict) else None
                    f.write(
                        "  {split}: intra_var_mean={intra} inter_mean={inter_mean} inter_min={inter_min} inter_max={inter_max} SNR={snr}\n".format(
                            split=split.capitalize(),
                            intra=fmt(stats.get("mean_intra_class_variance")),
                            inter_mean=fmt(inter_mean),
                            inter_min=fmt(inter_min),
                            inter_max=fmt(inter_max),
                            snr=fmt(snr_val),
                        )
                    )
                    per_class_var = stats.get("intra_class_variance") if isinstance(stats, dict) else None
                    if isinstance(per_class_var, dict):
                        f.write("    Per-class variance:\n")
                        for cls, var in per_class_var.items():
                            f.write(f"      {cls}: {fmt(var)}\n")
                if silhouette_path is not None:
                    f.write(f"  Test silhouette per-sample path: {silhouette_path}\n")
            if classification_samples is not None:
                f.write("Classification samples (test split, max 10):\n")
                correct_items = classification_samples.get("correct", [])
                wrong_items = classification_samples.get("misclassified", [])
                f.write(f"  Correctly classified ({len(correct_items)} shown):\n")
                for item in correct_items:
                    f.write(
                        f"    {item.path} [class={item.true_class}]\n"
                    )
                f.write(f"  Misclassified ({len(wrong_items)} shown):\n")
                for item in wrong_items:
                    f.write(
                        f"    {item.path} [pred={item.predicted_class}, true={item.true_class}]\n"
                    )
        print(f"[ACCEPT] Metrics saved to {path}")



def notify_telegram(message: str) -> None:
    if not TOKEN_TELEGRAM or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as exc:
        print(f"[WARN] Telegram notification failed: {exc}")


def parse_t_steps(raw: str) -> Tuple[int, ...]:
    items = [s.strip() for s in raw.split(",") if s.strip()]
    if not items:
        return (0,)
    return tuple(sorted({int(item) for item in items}))


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() in {"true", "1", "yes", "on"}


def parse_args() -> PipelineArgs:
    parser = argparse.ArgumentParser(description="ProtoLeakNet + SD2.1 training/evaluation pipeline.")
    parser.add_argument("--closed-root", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        choices=["conv", "resnet18", "resnet50", "resnet101", "efficientnet_b4", "vit"],
        default="conv",
    )
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--protos-per-class", type=int, default=6)
    parser.add_argument("--use-attention", type=str2bool, default=True)
    parser.add_argument("--metric", choices=["maha_diag", "proto_gmm", "kde", "llr_tied", "euclidean"], default="maha_diag")
    parser.add_argument("--var-floor", type=float, default=1e-3)
    parser.add_argument("--shrink", type=float, default=0.05)
    parser.add_argument("--gmm-components", type=str, default="2,3,4,6")
    parser.add_argument("--lse-temperature-learnable", type=str2bool, default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-steps", type=str, default="0")
    parser.add_argument("--sigma-normalize", type=str2bool, default=True)
    parser.add_argument("--highpass", type=str2bool, default=False)
    parser.add_argument("--latent-space", type=str2bool, default=True)
    parser.add_argument("--tau-quantile", type=float, default=0.95)
    parser.add_argument("--train-step", type=str, default="all", choices=["all", "step1", "step2", "step3"], help="Train only on a specific dataset step (step1/step2/step3) or 'all' for no filter")
    parser.add_argument("--calib", type=str2bool, default=True)
    parser.add_argument("--tta-flip", type=str2bool, default=False)
    parser.add_argument("--compressor", choices=["none", "pca", "dvae"], default="none")
    parser.add_argument("--pca", type=str2bool, default=None, help="Deprecated flag; use --compressor pca")
    parser.add_argument("--dvae", type=str2bool, default=None, help="Deprecated flag; use --compressor dvae")
    parser.add_argument("--pca-dim", type=int, default=None)
    parser.add_argument("--pca-sweep", type=str2bool, default=False)
    parser.add_argument("--pca-sweep-dims", type=str, default="16,32,48,64,96")
    parser.add_argument("--pca-whiten", type=str2bool, default=False)
    parser.add_argument("--dvae-latent-dim", type=int, default=128)
    parser.add_argument("--dvae-epochs", type=int, default=50)
    parser.add_argument("--dvae-batch-size", type=int, default=256)
    parser.add_argument("--dvae-lr", type=float, default=1e-3)
    parser.add_argument("--kde-bandwidth", type=float, default=0.5)
    parser.add_argument("--kde-bandwidths", type=str, default="0.1,0.2,0.4,0.8")
    parser.add_argument("--llr-shrink", type=float, default=0.1)
    parser.add_argument("--calibration-mode", choices=["zscore", "perclass_temp", "none"], default="zscore")
    parser.add_argument("--temperature-lr", type=float, default=1e-2)
    parser.add_argument("--temperature-steps", type=int, default=200)
    parser.add_argument("--time-dropout", type=float, default=0.3)
    parser.add_argument("--time-entropy-weight", type=float, default=0.1)
    parser.add_argument("--am-softmax", type=str2bool, default=False)
    parser.add_argument("--am-margin", type=float, default=0.35)
    parser.add_argument("--am-scale", type=float, default=30.0)
    parser.add_argument("--am-lambda-proto", type=float, default=0.5)
    parser.add_argument("--am-lambda-reg", type=float, default=0.0)
    parser.add_argument("--extra-proto-threshold", type=float, default=0.95)
    parser.add_argument("--extra-proto-count", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt")
    parser.add_argument("--run-name", type=str, default="run")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--make-class-list", type=str2bool, default=False)

    ns = parser.parse_args()
    gmm = tuple(int(s) for s in ns.gmm_components.split(",") if s.strip())
    compressor = ns.compressor.lower()
    if ns.pca is not None:
        compressor = "pca" if ns.pca else compressor
    if ns.dvae is not None:
        compressor = "dvae" if ns.dvae else compressor
    pca_dims = tuple(int(s) for s in ns.pca_sweep_dims.split(",") if s.strip())
    output_dir = ns.output_dir if isinstance(ns.output_dir, Path) else Path(ns.output_dir)
    return PipelineArgs(
        closed_root=ns.closed_root,
        backbone=ns.backbone,
        pretrained=bool(ns.pretrained),
        embed_dim=ns.embed_dim,
        protos_per_class=ns.protos_per_class,
        use_attention=bool(ns.use_attention),
        metric=ns.metric,
        var_floor=ns.var_floor,
        shrink=ns.shrink,
        gmm_components=gmm if gmm else (2, 3, 4, 6),
        lse_temperature_learnable=bool(ns.lse_temperature_learnable),
        batch_size=ns.batch_size,
        epochs=ns.epochs,
        lr=ns.lr,
        weight_decay=ns.weight_decay,
        device=ns.device,
        num_workers=ns.num_workers,
        seed=ns.seed,
        t_steps=parse_t_steps(ns.t_steps),
        sigma_normalize=bool(ns.sigma_normalize),
        highpass=bool(ns.highpass),
        latent_space=bool(ns.latent_space),
        tau_quantile=ns.tau_quantile,
        calib=bool(ns.calib),
        tta_flip=bool(ns.tta_flip),
        compressor=compressor,
        pca_dim=ns.pca_dim,
        pca_sweep=bool(ns.pca_sweep),
        pca_sweep_dims=pca_dims,
        pca_whiten=bool(ns.pca_whiten),
        dvae_latent_dim=ns.dvae_latent_dim,
        dvae_epochs=ns.dvae_epochs,
        dvae_batch_size=ns.dvae_batch_size,
        dvae_lr=ns.dvae_lr,
        kde_bandwidth=ns.kde_bandwidth,
        kde_bandwidths=tuple(float(s) for s in ns.kde_bandwidths.split(",") if s.strip()),
        llr_shrink=ns.llr_shrink,
        calibration_mode=ns.calibration_mode,
        temperature_lr=ns.temperature_lr,
        temperature_steps=ns.temperature_steps,
        time_dropout=ns.time_dropout,
        time_entropy_weight=ns.time_entropy_weight,
        am_softmax=bool(ns.am_softmax),
        am_margin=ns.am_margin,
        am_scale=ns.am_scale,
        am_lambda_proto=ns.am_lambda_proto,
        am_lambda_reg=ns.am_lambda_reg,
        extra_proto_auc_threshold=ns.extra_proto_threshold,
        extra_proto_count=ns.extra_proto_count,
        patience=ns.patience,
        checkpoint_name=ns.checkpoint_name,
        run_name=ns.run_name,
        output_dir=output_dir,
        train_step=ns.train_step,
        make_class_list=bool(ns.make_class_list),
    )


def main() -> None:
    args = parse_args()
    notify_telegram(
        f"ProtoLeak experiment starting (closed_root={args.closed_root}, metric={args.metric}, t_steps={args.t_steps})."
    )
    try:
        pipeline = ProtoLeakPipeline(args)
        pipeline.train()
        results = pipeline.evaluate()
    except Exception as exc:
        notify_telegram(f"ProtoLeak experiment failed: {exc}")
        raise
    else:
        notify_telegram(
            f"ProtoLeak experiment completed (macroAUC={results.closed_macro_auc:.4f})."
        )


if __name__ == "__main__":
    main()
