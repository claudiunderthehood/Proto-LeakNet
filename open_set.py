import argparse, json, math
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold

from proto_leaknet.model import LeakBackbone
from proto_leaknet.prototypes import ProtoHead
from heads.metrics_proto import LSEAggregator


# ------------------------------- Utils --------------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_fixed_channels(img, in_ch: int, alpha_policy: str):
    """
    For 4-channel latents, avoid cross-domain artifacts.
    - in_ch==4:
        keep        -> native RGBA
        zero        -> RGBA but force C4=0
        drop_usepad -> take RGB + append C4=0
    - in_ch!=4: force RGB and adapt (truncate/pad)
    """
    if in_ch == 4:
        if alpha_policy == "drop_usepad":
            if getattr(img, "mode", None) != "RGB":
                img = img.convert("RGB")
            t = TF.to_tensor(img)  # [3,H,W]
            pad = torch.zeros_like(t[:1])
            return torch.cat([t, pad], dim=0)
        else:
            if getattr(img, "mode", None) != "RGBA":
                img = img.convert("RGBA")
            t = TF.to_tensor(img)  # [4,H,W]
            if alpha_policy == "zero":
                t[3].zero_()
            return t
    else:
        if getattr(img, "mode", None) != "RGB":
            img = img.convert("RGB")
        t = TF.to_tensor(img)  # [3,H,W]
        c, h, w = t.shape
        if c == in_ch:
            return t
        if c > in_ch:
            return t[:in_ch]
        pad = torch.zeros(in_ch - c, h, w, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)


class FlatImageDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.root}")
        extensions = {ext.lower() for ext in IMG_EXTENSIONS}
        self.paths = sorted(
            p for p in self.root.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
        if not self.paths:
            raise FileNotFoundError(f"Couldn't find any images in {self.root}.")
        self.transform = transform
        self.loader = default_loader
        self.classes = ["__flat__"]
        self.targets = [0] * len(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        img = self.loader(str(path))
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def make_loader(root: Path, batch_size: int, num_workers: int, in_ch: int, alpha_policy: str):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        lambda img: _to_fixed_channels(img, in_ch, alpha_policy),
    ])
    root = Path(root)
    has_class_dirs = any(p.is_dir() for p in root.iterdir()) if root.is_dir() else False
    if has_class_dirs:
        ds = datasets.ImageFolder(root=str(root), transform=tf)
    else:
        ds = FlatImageDataset(root=root, transform=tf)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, pin_memory=True), ds.classes


def make_imagenet_loader(root: Path, batch_size: int, num_workers: int,
                         in_ch: int, alpha_policy: str):
    """ImageNet (all available images) used as the open set."""
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        lambda img: _to_fixed_channels(img, in_ch, alpha_policy),
    ])
    root = Path(root).expanduser()
    if not root.exists():
        raise RuntimeError(
            f"ImageNet not found in {root}. Manually copy the dataset (train/val) into this directory."
        )
    ds = None
    try:
        ds = datasets.ImageNet(root=str(root), split="val", transform=tf)
    except Exception as e:
        print(f"[WARN] torchvision.datasets.ImageNet failed: {e}. Falling back to ImageFolder.")
        val_dir = root / "val"
        img_root = val_dir if val_dir.is_dir() else root
        ds = datasets.ImageFolder(root=str(img_root), transform=tf)
        if len(ds) == 0:
            raise RuntimeError(
                f"ImageNet is unavailable in {root} and the ImageFolder fallback is empty. "
                "Download/copy the dataset or use --open-root."
            ) from e
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)
    return loader, ds.classes


@torch.no_grad()
def compute_input_stats(loader: DataLoader, max_batches: Optional[int] = None):
    """
    Per-channel mean/std over input tensors.
    CLOSED stats are used to normalize both CLOSED and OPEN.
    """
    n, s, s2 = 0, None, None
    seen = 0
    for i, (x, _) in enumerate(tqdm(loader, desc="Stats(closed)", leave=False)):
        x = x.float()  # [B,C,H,W] in [0,1]
        b, c, h, w = x.shape
        x_ = x.view(b, c, -1)       # [B,C,HW]
        sum_ = x_.sum(dim=(0, 2))   # [C]
        sum2 = (x_ ** 2).sum(dim=(0, 2))
        n += b * h * w
        s = sum_ if s is None else s + sum_
        s2 = sum2 if s2 is None else s2 + sum2
        seen += 1
        if max_batches is not None and seen >= max_batches:
            break
    mean = (s / n)
    var = (s2 / n) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    return mean.numpy(), std.numpy()


def apply_input_normalization(x: torch.Tensor, mean: np.ndarray, std: np.ndarray):
    """x: [B,C,H,W]  mean/std: [C]. Channels with std≈0 are bypassed."""
    m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    mask = (s > 1e-6).to(x.dtype)
    x = (x - m) * mask / torch.clamp(s, min=1e-6) + x * (1 - mask)
    return x


def diag_plot_channels(means_closed: np.ndarray, means_open: np.ndarray,
                       stds_closed: np.ndarray, stds_open: np.ndarray,
                       out_png: Path):
    ch = len(means_closed)
    xs = np.arange(ch)
    plt.figure(figsize=(6, 4))
    plt.subplot(2, 1, 1)
    plt.title("Per-channel mean (pre-normalization)")
    plt.plot(xs, means_closed, "o-", label="Closed")
    plt.plot(xs, means_open, "o--", label="Open")
    plt.xticks(xs, [f"C{i+1}" for i in xs])
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("Per-channel std (pre-normalization)")
    plt.plot(xs, stds_closed, "o-", label="Closed")
    plt.plot(xs, stds_open, "o--", label="Open")
    plt.xticks(xs, [f"C{i+1}" for i in xs])
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=160)
    plt.close()


@torch.no_grad()
def collect_embeddings(backbone: LeakBackbone, loader: DataLoader,
                       use_attention: bool, device: torch.device,
                       norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                       return_attention: bool = False):
    backbone.eval()
    Z, y = [], []
    W: Optional[List[Optional[torch.Tensor]]] = [] if return_attention else None
    for x, labels in tqdm(loader, desc=f"Embedding(att={'ON' if use_attention else 'OFF'})", leave=False):
        x = x.to(device, non_blocking=True)
        if norm_stats is not None:
            x = apply_input_normalization(x, *norm_stats)
        h, w = backbone(x)  # h: [B,D], w: [B,D] or None
        if use_attention and (w is not None):
            h = h * w
        Z.append(h.cpu())
        y.append(labels)
        if return_attention and W is not None:
            W.append(w.cpu() if w is not None else None)

    Z_all = torch.cat(Z).numpy()
    y_all = torch.cat(y).numpy()
    if not return_attention or W is None:
        return Z_all, y_all

    assert W is not None
    if all(w is None for w in W):
        W_all = None
    elif any(w is None for w in W):
        raise RuntimeError("Inconsistent attention availability within the same loader.")
    else:
        W_all = torch.cat([w for w in W if w is not None]).numpy()
    return Z_all, y_all, W_all


def _eer_from_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    idx = np.nanargmin(np.abs(1 - tpr - fpr))
    eer = 0.5 * (fpr[idx] + (1 - tpr[idx]))
    return eer


def _fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """
    False Positive Rate at a given True Positive Rate.
    Chooses the minimum FPR whose TPR is >= target_tpr.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = tpr >= target_tpr
    if not np.any(mask):
        return float(fpr[-1])
    return float(np.min(fpr[mask]))


def label_shuffle_auc(y_true: np.ndarray, scores: np.ndarray, n_trials: int = 5, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_trials):
        y_shuf = rng.permutation(y_true)
        vals.append(roc_auc_score(y_shuf, scores))
    return float(np.mean(vals))


def kde_auc_kfold(Zc: np.ndarray, Zo: np.ndarray, bandwidth: float = 0.4, n_splits: int = 5, seed: int = 42):
    """AUROC/EER while avoiding in-sample bias on CLOSED via K-fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    log_p_c = np.empty(len(Zc), dtype=float)
    for tr, te in kf.split(Zc):
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(Zc[tr])
        log_p_c[te] = kde.score_samples(Zc[te])
    kde_all = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(Zc)
    log_p_o = kde_all.score_samples(Zo)

    y_true = np.concatenate([np.ones(len(Zc)), np.zeros(len(Zo))])
    scores = np.concatenate([log_p_c, log_p_o])
    auc = roc_auc_score(y_true, scores)
    eer = _eer_from_roc(y_true, scores)
    return auc, eer, log_p_c, log_p_o, y_true, scores


def select_bandwidth_cv(Zc: np.ndarray, grid: List[float], n_splits: int = 5, seed: int = 42) -> float:
    """Select bandwidth by maximizing out-of-fold log-likelihood on CLOSED."""
    best_bw, best_ll = None, -math.inf
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for bw in grid:
        fold_ll = 0.0
        for tr, te in kf.split(Zc):
            kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(Zc[tr])
            fold_ll += kde.score_samples(Zc[te]).mean()
        if fold_ll > best_ll:
            best_ll, best_bw = fold_ll, bw
    return float(best_bw)


def auc_vs_bandwidth(Zc: np.ndarray, Zo: np.ndarray, grid: List[float], n_splits: int, seed: int):
    xs, ys = [], []
    for bw in grid:
        auc, _, _, _, _, _ = kde_auc_kfold(Zc, Zo, bandwidth=float(bw), n_splits=n_splits, seed=seed)
        xs.append(float(bw)); ys.append(float(auc))
    return np.array(xs), np.array(ys)


def plot_bw_curve(xs: np.ndarray, ys: np.ndarray, out_png: Path):
    plt.figure(figsize=(5, 3.2))
    plt.plot(xs, ys, "o-")
    plt.xlabel("Bandwidth")
    plt.ylabel("AUROC")
    plt.title("KDE AUROC vs Bandwidth")
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=160)
    plt.close()


def score_hist_plot(scores_closed: np.ndarray, scores_open: np.ndarray,
                    auc: float, eer: float, title: str, out_png: Path):
    plt.figure(figsize=(6, 4))
    plt.hist(scores_closed, bins=60, alpha=0.6, label="Closed")
    plt.hist(scores_open, bins=60, alpha=0.6, label="Open")
    plt.legend()
    plt.title(f"{title}\nAUC={auc:.3f}, EER={eer:.3f}")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=160)
    plt.close()


def compute_overlap_coefficient(scores_closed: np.ndarray, scores_open: np.ndarray, num_bins: int = 256) -> float:
    """Approximate overlap coefficient between two 1D score distributions via histogram density."""
    data_min = float(min(scores_closed.min(), scores_open.min()))
    data_max = float(max(scores_closed.max(), scores_open.max()))
    if data_max <= data_min + 1e-12:
        return 1.0
    bins = np.linspace(data_min, data_max, num_bins + 1)
    hist_c, _ = np.histogram(scores_closed, bins=bins, density=True)
    hist_o, _ = np.histogram(scores_open, bins=bins, density=True)
    overlap = np.minimum(hist_c, hist_o)
    return float(np.sum(overlap * np.diff(bins)))


def score_cdf_plot(scores_closed: np.ndarray, scores_open: np.ndarray,
                   overlap: float, title: str, out_png: Path):
    """Plot empirical CDFs of generic scores for closed and open sets."""
    grid_min = float(min(scores_closed.min(), scores_open.min()))
    grid_max = float(max(scores_closed.max(), scores_open.max()))
    if grid_max <= grid_min + 1e-12:
        grid_max = grid_min + 1.0
    grid = np.linspace(grid_min, grid_max, 512)
    c_sorted = np.sort(scores_closed)
    o_sorted = np.sort(scores_open)
    c_cdf = np.searchsorted(c_sorted, grid, side="right") / c_sorted.size
    o_cdf = np.searchsorted(o_sorted, grid, side="right") / o_sorted.size
    plt.figure(figsize=(6, 4))
    plt.plot(grid, c_cdf, label="Closed")
    plt.plot(grid, o_cdf, label="Open")
    plt.title(f"{title} CDF (Overlap={overlap:.3f})")
    plt.xlabel("Score")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=160)
    plt.close()


def evaluate_binary_scores(method: str,
                           scores_closed: np.ndarray,
                           scores_open: np.ndarray,
                           *,
                           report_dir: Path,
                           tag: str,
                           args,
                           seed: int,
                           precomputed_auc: Optional[float] = None,
                           precomputed_eer: Optional[float] = None,
                           y_true: Optional[np.ndarray] = None,
                           scores_all: Optional[np.ndarray] = None) -> Dict[str, Optional[float]]:
    """Compute AUROC/EER/FPR@95/overlap + plots + optional label-shuffle sanity."""
    if y_true is None or scores_all is None:
        y_true = np.concatenate([np.ones(len(scores_closed)), np.zeros(len(scores_open))])
        scores_all = np.concatenate([scores_closed, scores_open])
        auc = roc_auc_score(y_true, scores_all)
        eer = _eer_from_roc(y_true, scores_all)
    else:
        auc = float(precomputed_auc) if precomputed_auc is not None else roc_auc_score(y_true, scores_all)
        eer = float(precomputed_eer) if precomputed_eer is not None else _eer_from_roc(y_true, scores_all)

    fpr95 = _fpr_at_tpr(y_true, scores_all, target_tpr=0.95)
    overlap = compute_overlap_coefficient(scores_closed, scores_open)
    auc_shuf = None
    if args.label_shuffle:
        auc_shuf = label_shuffle_auc(y_true, scores_all, n_trials=5, seed=seed)

    title = method.upper()
    score_hist_plot(scores_closed, scores_open, auc, eer,
                    title=f"{title} scores", out_png=report_dir / f"{tag}_{method}_scores.png")
    score_cdf_plot(scores_closed, scores_open, overlap,
                   title=title, out_png=report_dir / f"{tag}_{method}_cdf.png")

    return dict(auc=float(auc), eer=float(eer), fpr95=float(fpr95), overlap=float(overlap),
                auc_shuffle=(float(auc_shuf) if auc_shuf is not None else None))


def _fit_mahal_model(Z: np.ndarray, reg: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = Z.mean(axis=0)
    if Z.shape[0] <= 1:
        cov = np.eye(Z.shape[1], dtype=np.float64)
    else:
        cov = np.cov(Z, rowvar=False)
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[0]
    trace = float(np.trace(cov)) if d > 0 else 1.0
    reg_scale = float(reg) * trace / max(d, 1)
    cov = cov + np.eye(d, dtype=np.float64) * (reg_scale + 1e-6)
    inv = np.linalg.pinv(cov)
    return mean.astype(np.float64), inv


def _mahal_scores(Z: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    diff = Z.astype(np.float64) - mean
    proj = diff @ inv_cov
    return -0.5 * np.sum(proj * diff, axis=1)


def mahal_auc_kfold(Zc: np.ndarray, Zo: np.ndarray, reg: float,
                    n_splits: int, seed: int):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores_closed = np.empty(len(Zc), dtype=np.float64)
    for tr, te in kf.split(Zc):
        mean, inv = _fit_mahal_model(Zc[tr], reg=reg)
        scores_closed[te] = _mahal_scores(Zc[te], mean, inv)
    mean_all, inv_all = _fit_mahal_model(Zc, reg=reg)
    scores_open = _mahal_scores(Zo, mean_all, inv_all)
    y_true = np.concatenate([np.ones(len(Zc)), np.zeros(len(Zo))])
    scores_all = np.concatenate([scores_closed, scores_open])
    auc = roc_auc_score(y_true, scores_all)
    eer = _eer_from_roc(y_true, scores_all)
    return auc, eer, scores_closed, scores_open, y_true, scores_all


@torch.no_grad()
def proto_logits_from_numpy(proto_head: ProtoHead,
                            embeddings: np.ndarray,
                            attention: Optional[np.ndarray],
                            device: torch.device,
                            lse_agg: Optional[LSEAggregator] = None,
                            batch_size: int = 2048) -> np.ndarray:
    proto_head.eval()
    if lse_agg is not None:
        lse_agg.eval()
    logits = []
    N = embeddings.shape[0]
    if N == 0:
        return np.empty((0, proto_head.K), dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        h = torch.from_numpy(embeddings[start:end]).to(device)
        if attention is not None:
            w = torch.from_numpy(attention[start:end]).to(device)
        else:
            w = None
        dists = proto_head.distances(h, w)
        if lse_agg is not None:
            cls_scores = lse_agg(dists)
        else:
            cls_scores = -dists.min(dim=-1).values
        logits.append(cls_scores.detach().cpu().numpy())
    return np.concatenate(logits, axis=0)


def energy_confidence(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return -energy, which acts as confidence for closed-vs-open."""
    T = float(max(temperature, 1e-6))
    scaled = logits / T
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    lse = max_scaled + np.log(np.exp(scaled - max_scaled).sum(axis=1, keepdims=True))
    conf = (T * lse).squeeze(1)
    return conf.astype(np.float64)


def compute_umap_coords(Zc: np.ndarray, Zo: np.ndarray):
    try:
        import umap
    except Exception as e:
        print(f"[WARN] UMAP skipped (import): {e}")
        return None
    try:
        reducer = umap.UMAP(n_components=2, random_state=42)
        X = np.concatenate([Zc, Zo])
        Y = reducer.fit_transform(X)
        nc = len(Zc)
        return Y[:nc], Y[nc:]
    except Exception as e:
        print(f"[WARN] UMAP failed during fit: {e}")
        return None


def plot_umap_closed_open(Yc: np.ndarray, Yo: np.ndarray, out_png: Path):
    plt.figure(figsize=(7, 6))
    plt.scatter(Yc[:, 0], Yc[:, 1], s=6, alpha=0.4, label="Closed")
    plt.scatter(Yo[:, 0], Yo[:, 1], s=6, alpha=0.4, label="Open")
    plt.legend()
    plt.title("UMAP projection")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=160)
    plt.close()
    print(f"[SAVE] UMAP -> {out_png}")


def plot_umap_scores(Yc: np.ndarray, Yo: np.ndarray,
                     scores_closed: np.ndarray, scores_open: np.ndarray,
                     out_png: Path, title: str):
    plt.figure(figsize=(7, 6))
    all_scores = np.concatenate([scores_closed, scores_open])
    if np.allclose(all_scores.max(), all_scores.min()):
        vmin = all_scores.min() - 0.5
        vmax = all_scores.max() + 0.5
    else:
        vmin = np.percentile(all_scores, 2)
        vmax = np.percentile(all_scores, 98)
        if vmax <= vmin:
            vmax = vmin + 1e-6
    sc_closed = plt.scatter(Yc[:, 0], Yc[:, 1], s=6, alpha=0.6,
                            c=scores_closed, cmap="viridis", vmin=vmin, vmax=vmax, label="Closed")
    plt.scatter(Yo[:, 0], Yo[:, 1], s=6, alpha=0.6,
                c=scores_open, cmap="viridis", vmin=vmin, vmax=vmax, marker="x", label="Open")
    plt.colorbar(sc_closed, label="Score")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=160)
    plt.close()
    print(f"[SAVE] UMAP -> {out_png}")


@dataclass
class EncodedSets:
    Zc: np.ndarray
    Zo: np.ndarray
    Wc: Optional[np.ndarray] = None
    Wo: Optional[np.ndarray] = None


def encode_with_mode(backbone: LeakBackbone,
                     closed_loader: DataLoader,
                     open_loader: DataLoader,
                     mode: str,
                     device: torch.device,
                     norm_stats: Optional[Tuple[np.ndarray, np.ndarray]],
                     return_attention: bool = False):
    """
    mode ∈ {'both_off','both_on','dual'}
      - both_off: closed OFF, open OFF
      - both_on : closed ON , open ON
      - dual    : closed ON , open OFF (original setup)
    """
    assert mode in {"both_off", "both_on", "dual"}
    att_orig = getattr(backbone, "att", None)

    def _set_att(flag: bool):
        if flag:
            backbone.att = att_orig
        else:
            backbone.att = None

    if mode == "both_off":
        _set_att(False)
        closed = collect_embeddings(backbone, closed_loader, use_attention=False,
                                    device=device, norm_stats=norm_stats, return_attention=return_attention)
        open_ = collect_embeddings(backbone, open_loader, use_attention=False,
                                   device=device, norm_stats=norm_stats, return_attention=return_attention)
    elif mode == "both_on":
        if att_orig is None:
            raise RuntimeError("Backbone has no attention module for 'both_on'.")
        _set_att(True)
        closed = collect_embeddings(backbone, closed_loader, use_attention=True,
                                    device=device, norm_stats=norm_stats, return_attention=return_attention)
        open_ = collect_embeddings(backbone, open_loader, use_attention=True,
                                   device=device, norm_stats=norm_stats, return_attention=return_attention)
    else:  # dual
        if att_orig is None:
            raise RuntimeError("Backbone has no attention module for 'dual'.")
        _set_att(True)
        closed = collect_embeddings(backbone, closed_loader, use_attention=True,
                                    device=device, norm_stats=norm_stats, return_attention=return_attention)
        _set_att(False)
        open_ = collect_embeddings(backbone, open_loader, use_attention=False,
                                   device=device, norm_stats=norm_stats, return_attention=return_attention)

    backbone.att = att_orig

    if return_attention:
        Zc, _, Wc = closed
        Zo, _, Wo = open_
    else:
        Zc, _ = closed
        Zo, _ = open_
        Wc = Wo = None
    return EncodedSets(Zc=Zc, Zo=Zo, Wc=Wc, Wo=Wo)


# ------------------------------- Main ---------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--closed-root", type=Path, required=True)
    p.add_argument("--open-root", type=Path, required=False, default=None,
                   help="Required unless --image-net is used.")
    p.add_argument("--image-net", action="store_true",
                   help="Use all ImageNet images (val split or ImageFolder structure) from datasets/ImageNet as the open set.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--report-dir", type=Path, default=Path("reports/metric_learning"))
    p.add_argument("--bandwidth", type=float, default=None, help="If not provided, select it via CV")
    p.add_argument("--bw-grid", type=str, default="0.1,0.2,0.3,0.4,0.6,0.8,1.0")
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--compare-modes", action="store_true",
                   help="Compare both_off, both_on, and dual. If omitted, only use 'dual'.")
    p.add_argument("--diag", action="store_true", help="Save per-channel diagnostics (pre-normalization)")
    p.add_argument("--alpha-policy", type=str, default="zero",
                   choices=["keep", "zero", "drop_usepad"],
                   help="4th-channel handling when in_ch==4")

    # New sanity checks
    p.add_argument("--seed-sweep", type=int, default=1,
                   help="Number of independent seeds for mean±std (>=1).")
    p.add_argument("--alpha-sweep", action="store_true",
                   help="Evaluate all alpha policies {keep,zero,drop_usepad}.")
    p.add_argument("--label-shuffle", action="store_true",
                   help="AUROC with shuffled labels (expected ≈ 0.5).")
    p.add_argument("--bw-curve", action="store_true",
                   help="Plot AUROC vs bandwidth over the --bw-grid values.")
    p.add_argument("--ablation", action="store_true",
                   help="Compare KDE vs Mahalanobis vs Energy (dual mode).")
    p.add_argument("--mahal-reg", type=float, default=1e-3,
                   help="Diagonal regularization for Mahalanobis.")
    p.add_argument("--energy-temperature", type=float, default=1.0,
                   help="Temperature for the Energy score.")
    args = p.parse_args()

    if not args.image_net and args.open_root is None:
        p.error("--open-root is required unless --image-net is used.")

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.report_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone_state = ckpt["backbone"]

    # Detect input channels & embedding dim
    conv_key = next(k for k in backbone_state.keys() if "encoder.base.conv1.weight" in k)
    head_key = next(k for k in backbone_state.keys() if "head.weight" in k)
    in_ch = backbone_state[conv_key].shape[1]
    embed_dim = backbone_state[head_key].shape[0]
    print(f"[INFO] Detected input channels={in_ch}, embed_dim={embed_dim}")

    # Build backbone
    backbone = LeakBackbone(
        in_ch=in_ch,
        embed_dim=embed_dim,
        use_attention=True,
        encoder_type="resnet18",
        resnet_pretrained=False
    )
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.to(dev)

    proto_head = None
    lse_agg = None
    if args.ablation:
        proto_state = ckpt.get("proto_head")
        if proto_state is None or "protos" not in proto_state:
            raise RuntimeError("Checkpoint is missing 'proto_head', required for the Energy ablation.")
        protos = proto_state["protos"]
        num_classes, protos_per_class, proto_dim = protos.shape
        if proto_dim != embed_dim:
            raise RuntimeError(f"embed_dim mismatch: backbone={embed_dim}, proto_head={proto_dim}.")
        proto_head = ProtoHead(
            num_classes=num_classes,
            embed_dim=embed_dim,
            protos_per_class=protos_per_class,
            agg="min",
        ).to(dev)
        proto_head.load_state_dict(proto_state, strict=True)
        proto_head.eval()
        if "lse_agg" in ckpt and ckpt["lse_agg"] is not None:
            lse_agg = LSEAggregator().to(dev)
            lse_agg.load_state_dict(ckpt["lse_agg"])
            lse_agg.eval()

    def build_open_loader(alpha_policy: str):
        if args.image_net:
            imagenet_path = Path("datasets/ImageNet")
            print(f"[INFO] Using ALL ImageNet images from {imagenet_path} as the open set.")
            return make_imagenet_loader(imagenet_path, args.batch_size, args.num_workers,
                                        in_ch, alpha_policy)
        assert args.open_root is not None
        return make_loader(args.open_root, args.batch_size, args.num_workers, in_ch, alpha_policy)

    # Load data (first instance with the default policy; rebuilt if --alpha-sweep is used)
    closed_loader, _ = make_loader(args.closed_root, args.batch_size, args.num_workers, in_ch, args.alpha_policy)
    open_loader, _   = build_open_loader(args.alpha_policy)

    # -------- Input diagnostics and normalization (closed-driven) --------
    closed_mean, closed_std = compute_input_stats(closed_loader)
    norm_stats = (closed_mean, closed_std)
    print(f"[STATS] closed mean: {np.round(closed_mean,4)}")
    print(f"[STATS] closed std : {np.round(closed_std,4)}")

    if args.diag:
        open_mean, open_std = compute_input_stats(open_loader)
        print(f"[STATS] open   mean: {np.round(open_mean,4)}")
        print(f"[STATS] open   std : {np.round(open_std,4)}")
        diag_plot_channels(closed_mean, open_mean, closed_std, open_std,
                           out_png=args.report_dir / "diagnostics_channels.png")

    results: Dict[str, Dict[str, List[dict]]] = {}
    if args.ablation and args.compare_modes:
        print("[WARN] --ablation forces 'dual' mode; ignoring --compare-modes.")
    modes = ("dual",) if args.ablation or not args.compare_modes else ("both_off", "both_on", "dual")

    def accumulate(res_dict, key, method, val):
        if key not in res_dict:
            res_dict[key] = {}
        if method not in res_dict[key]:
            res_dict[key][method] = []
        res_dict[key][method].append(val)

    alpha_policies = ("keep", "zero", "drop_usepad") if args.alpha_sweep else (args.alpha_policy,)

    def run_one(mode: str, alpha_policy: Optional[str] = None, seed_override: Optional[int] = None):
        if alpha_policy is not None:
            print(f"[ALPHA] policy = {alpha_policy}")
        if seed_override is not None:
            set_seed(seed_override)

        enc = encode_with_mode(backbone, closed_loader, open_loader,
                               mode=mode, device=dev, norm_stats=norm_stats,
                               return_attention=args.ablation)
        tag = f"{mode}" + (f"_{alpha_policy}" if alpha_policy else "") + (f"_seed{seed_override}" if seed_override is not None else "")
        method_outputs: Dict[str, dict] = {}

        # Bandwidth selection / fixed
        if args.bandwidth is None:
            grid = [float(x) for x in args.bw_grid.split(",") if x.strip()]
            bw = select_bandwidth_cv(enc.Zc, grid=grid, n_splits=args.kfold, seed=(seed_override or args.seed))
            print(f"[BW] selected via CV: {bw}")
        else:
            bw = float(args.bandwidth)
            print(f"[BW] fixed: {bw}")

        auc, eer, log_c, log_o, y_true, scores = kde_auc_kfold(
            enc.Zc, enc.Zo, bandwidth=bw, n_splits=args.kfold, seed=(seed_override or args.seed)
        )

        metrics_kde = evaluate_binary_scores(
            "kde", log_c, log_o,
            report_dir=args.report_dir, tag=tag, args=args, seed=(seed_override or args.seed),
            precomputed_auc=auc, precomputed_eer=eer, y_true=y_true, scores_all=scores
        )
        print(f"[RESULT][{mode}][KDE] AUROC={metrics_kde['auc']:.4f} | EER={metrics_kde['eer']:.4f} | FPR@95={metrics_kde['fpr95']:.4f} | Overlap={metrics_kde['overlap']:.4f}")
        if metrics_kde.get("auc_shuffle") is not None:
            print(f"[SANITY][{mode}][KDE] AUROC shuffle ≈ {metrics_kde['auc_shuffle']:.3f}")
        method_outputs["kde"] = {**metrics_kde, "bandwidth": float(bw)}
        scores_per_method = {"kde": (log_c, log_o)}

        # Bandwidth curve
        if args.bw_curve:
            grid = [float(x) for x in args.bw_grid.split(",") if x.strip()]
            xs, ys = auc_vs_bandwidth(enc.Zc, enc.Zo, grid=grid, n_splits=args.kfold, seed=(seed_override or args.seed))
            plot_bw_curve(xs, ys, out_png=args.report_dir / f"{tag}_bw_curve.png")

        # Save NPZ
        npz_path = args.report_dir / f"{tag}_embeddings.npz"
        npz_payload = dict(
            Z_closed=enc.Zc, Z_open=enc.Zo,
            logp_closed=log_c, logp_open=log_o,
            auc=metrics_kde["auc"], eer=metrics_kde["eer"], fpr95=metrics_kde["fpr95"],
            auc_shuffle=(metrics_kde["auc_shuffle"] if metrics_kde["auc_shuffle"] is not None else np.nan),
            bandwidth=bw,
            checkpoint=str(args.checkpoint),
            mode=mode,
            seed=(seed_override or args.seed), kfold=args.kfold,
            input_mean=norm_stats[0], input_std=norm_stats[1],
            alpha_policy=(alpha_policy or args.alpha_policy),
            overlap=metrics_kde["overlap"],
        )

        if args.ablation:
            maha_auc, maha_eer, maha_c, maha_o, maha_y, maha_scores = mahal_auc_kfold(
                enc.Zc, enc.Zo, reg=args.mahal_reg,
                n_splits=args.kfold, seed=(seed_override or args.seed)
            )
            metrics_maha = evaluate_binary_scores(
                "mahalanobis", maha_c, maha_o,
                report_dir=args.report_dir, tag=tag, args=args, seed=(seed_override or args.seed),
                precomputed_auc=maha_auc, precomputed_eer=maha_eer,
                y_true=maha_y, scores_all=maha_scores
            )
            print(f"[RESULT][{mode}][MAHAL] AUROC={metrics_maha['auc']:.4f} | EER={metrics_maha['eer']:.4f} | FPR@95={metrics_maha['fpr95']:.4f} | Overlap={metrics_maha['overlap']:.4f}")
            method_outputs["mahalanobis"] = metrics_maha
            scores_per_method["mahalanobis"] = (maha_c, maha_o)
            npz_payload.update(
                mahal_scores_closed=maha_c,
                mahal_scores_open=maha_o,
                mahal_fpr95=metrics_maha["fpr95"],
            )

            if proto_head is None:
                raise RuntimeError("ProtoHead missing: unable to compute Energy.")
            logits_closed = proto_logits_from_numpy(proto_head, enc.Zc, enc.Wc, device=dev, lse_agg=lse_agg)
            logits_open = proto_logits_from_numpy(proto_head, enc.Zo, enc.Wo, device=dev, lse_agg=lse_agg)
            energy_closed = energy_confidence(logits_closed, temperature=args.energy_temperature)
            energy_open = energy_confidence(logits_open, temperature=args.energy_temperature)
            metrics_energy = evaluate_binary_scores(
                "energy", energy_closed, energy_open,
                report_dir=args.report_dir, tag=tag, args=args, seed=(seed_override or args.seed)
            )
            print(f"[RESULT][{mode}][ENERGY] AUROC={metrics_energy['auc']:.4f} | EER={metrics_energy['eer']:.4f} | FPR@95={metrics_energy['fpr95']:.4f} | Overlap={metrics_energy['overlap']:.4f}")
            method_outputs["energy"] = {**metrics_energy, "temperature": args.energy_temperature}
            scores_per_method["energy"] = (energy_closed, energy_open)
            npz_payload.update(
                energy_scores_closed=energy_closed,
                energy_scores_open=energy_open,
                energy_temperature=args.energy_temperature,
                energy_fpr95=metrics_energy["fpr95"],
            )

        np.savez(str(npz_path), **npz_payload)
        print(f"[SAVE] Embeddings + metrics -> {npz_path}")

        if args.plot or args.compare_modes or args.seed_sweep > 1:
            coords = compute_umap_coords(enc.Zc, enc.Zo)
            if coords is not None:
                Yc, Yo = coords
                plot_umap_closed_open(Yc, Yo, out_png=args.report_dir / f"umap_{tag}.png")
                if args.ablation:
                    for method_name, (sc_closed, sc_open) in scores_per_method.items():
                        plot_umap_scores(
                            Yc, Yo,
                            scores_closed=sc_closed,
                            scores_open=sc_open,
                            out_png=args.report_dir / f"umap_{tag}_{method_name}.png",
                            title=f"UMAP colored by {method_name.upper()} score"
                        )

        return method_outputs

    # Loop: alpha policies × modes × seed sweep
    for ap in alpha_policies:
        # Rebuild loaders with the requested policy to avoid mismatches
        closed_loader, _ = make_loader(args.closed_root, args.batch_size, args.num_workers, in_ch, ap)
        open_loader, _   = build_open_loader(ap)

        # Recompute closed stats for consistency with the selected policy
        closed_mean, closed_std = compute_input_stats(closed_loader)
        norm_stats = (closed_mean, closed_std)

        for mode in modes:
            key = f"{mode}_{ap}"
            for s in range(args.seed_sweep):
                seed_i = args.seed + s
                out = run_one(mode=mode, alpha_policy=ap, seed_override=seed_i)
                for method_name, metrics in out.items():
                    accumulate(results, key, method_name, metrics)

    # Mean±std summary
    summary = {}
    prefix_required = args.ablation or any(len(methods) > 1 for methods in results.values())
    for key, methods in results.items():
        entry = {}
        for method, vals in methods.items():
            aucs = np.array([v["auc"] for v in vals], float)
            eers = np.array([v["eer"] for v in vals], float)
            fprs = np.array([v["fpr95"] for v in vals], float)
            overlaps = np.array([v["overlap"] for v in vals], float)
            aucs_shuf = np.array([v.get("auc_shuffle") for v in vals if v.get("auc_shuffle") is not None], float)
            prefix = f"{method}_" if prefix_required else ""
            entry[f"{prefix}auc_mean"] = float(aucs.mean())
            entry[f"{prefix}auc_std"] = float(aucs.std())
            entry[f"{prefix}eer_mean"] = float(eers.mean())
            entry[f"{prefix}eer_std"] = float(eers.std())
            entry[f"{prefix}fpr95_mean"] = float(fprs.mean())
            entry[f"{prefix}fpr95_std"] = float(fprs.std())
            entry[f"{prefix}overlap_mean"] = float(overlaps.mean())
            entry[f"{prefix}overlap_std"] = float(overlaps.std())
            entry[f"{prefix}auc_shuffle_mean"] = (float(aucs_shuf.mean()) if aucs_shuf.size else None)
        summary[key] = entry

    with open(args.report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Summary -> {args.report_dir/'summary.json'}")


if __name__ == "__main__":
    main()
