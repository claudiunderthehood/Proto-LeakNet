from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_embed_t(t: torch.Tensor, dims: int = 16) -> torch.Tensor:
    """Fourier embedding for integer/float t -> R^{2*dims} (float32).
    Accepts t of shape [B] or [B,1]. Always returns float features to feed Linear layers.
    """
    if t.ndim == 1:
        t = t[:, None]
    device = t.device
    t_f = t.to(torch.float32)
    freqs = torch.exp(torch.linspace(np.log(1.0), np.log(1000.0), dims, device=device, dtype=torch.float32))
    angles = t_f * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return emb.to(torch.float32)


class FiLM(nn.Module):
    def __init__(self, in_dim: int, feat_dim: int):
        super().__init__()
        self.aff = nn.Sequential(
            nn.Linear(in_dim, max(16, in_dim)), nn.ReLU(),
            nn.Linear(max(16, in_dim), 2 * feat_dim)
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.aff(cond)
        g, b = gamma_beta.chunk(2, dim=1)
        return h * (1.0 + g) + b


class LeakDVAE(nn.Module):
    """Denoising VAE over 1D embedding vectors (replacing PCA).

    Encoder/decoder are small MLP stacks. Optional FiLM conditioning on t via
    Fourier embedding.
    """

    def __init__(self, in_dim: Optional[int], z_dim: int = 128, use_t: bool = False, hidden: int = 256):
        super().__init__()
        self.in_dim = in_dim  # may be None before first fit
        self.z_dim = int(z_dim)
        self.use_t = bool(use_t)
        self.hidden = int(hidden)

        # Lazy-init modules once in_dim is known
        self.enc = None
        self.mu = None
        self.logvar = None
        self.dec = None
        self.film = None

    def _init_modules(self):
        assert self.in_dim is not None and self.in_dim > 0
        self.enc = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
        )
        if self.use_t:
            self.film = FiLM(in_dim=32, feat_dim=self.hidden)
        self.mu = nn.Linear(self.hidden, self.z_dim)
        self.logvar = nn.Linear(self.hidden, self.z_dim)
        self.dec = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.in_dim),
        )

    def to_device(self, device: str = "cpu"):
        return self.to(torch.device(device))

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.enc is None:
            self.in_dim = x.shape[1]
            self._init_modules()
            self.to(x.device)
        h = self.enc(x)
        if self.use_t and t is not None:
            emb = fourier_embed_t(t.to(x.device), dims=16)
            h = self.film(h, emb)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_hat = self.dec(z)
        return x_hat, mu, logvar, z


@dataclass
class DVAELossWeights:
    rec: float = 1.0
    kl: float = 0.5
    tcons: float = 1.0
    ctr: float = 0.2
    spec: float = 0.1


class DVAETrainer:
    def __init__(self, model: LeakDVAE, device: str = "cpu", beta_max: float = 0.5):
        self.model = model
        self.device = torch.device(device)
        self.w = DVAELossWeights()
        self.beta_max = float(beta_max)
        self.history: Dict[str, list] = {"rec": [], "kl": [], "tcons": [], "ctr": [], "spec": [], "total": []}

    # --- corruption ---
    def corrupt(self, x: torch.Tensor) -> torch.Tensor:
        # small gaussian noise + random drop + DCT/FFT jitter
        noise = 0.01 * torch.randn_like(x)
        if x.shape[1] > 8:
            # random feature dropout mask
            drop_mask = (torch.rand_like(x) > 0.98).float()
        else:
            drop_mask = 0.0
        x_noisy = x + noise - drop_mask * x
        # FFT magnitude jitter (force float32 to avoid fp16 cuFFT constraints)
        x_noisy_f32 = x_noisy.to(torch.float32)
        Xf = torch.fft.rfft(x_noisy_f32, dim=1)
        mag = torch.abs(Xf)
        phase = torch.angle(Xf)
        # small mag jitter and phase jitter
        mag = mag * (1.0 + 0.02 * torch.randn_like(mag))
        phase = phase + 0.02 * torch.randn_like(phase)
        Xf_j = mag * torch.exp(1j * phase)
        x_j = torch.fft.irfft(Xf_j, n=x_noisy_f32.shape[1], dim=1)
        return x_j.to(x.dtype)

    # --- losses ---
    @staticmethod
    def _kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    @staticmethod
    def _spec_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 to ensure rFFT runs in supported precision for non-power-of-two sizes
        Xh = torch.fft.rfft(x_hat.to(torch.float32), dim=1)
        X = torch.fft.rfft(x.to(torch.float32), dim=1)
        return F.l1_loss(torch.abs(Xh), torch.abs(X), reduction='none').mean(dim=1)

    @staticmethod
    def _t_consistency(mu: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        """Within-batch consistency: pull together embeddings of same class."""
        if y is None:
            return mu.new_zeros(mu.shape[0], dtype=mu.dtype)
        loss = mu.new_zeros(mu.shape[0], dtype=mu.dtype)
        for c in torch.unique(y):
            idx = torch.where(y == c)[0]
            if len(idx) < 2:
                continue
            m = mu[idx]
            center = m.mean(dim=0, keepdim=True)
            val = ((m - center) ** 2).sum(dim=1)
            loss[idx] = val.to(loss.dtype)
        return loss

    @staticmethod
    def _info_nce(mu: torch.Tensor, y: Optional[torch.Tensor], temp: float = 0.2) -> torch.Tensor:
        if y is None:
            return mu.new_zeros(mu.shape[0], dtype=mu.dtype)
        mu_n = F.normalize(mu, dim=1)
        sims = (mu_n @ mu_n.t()) / temp
        B = mu.shape[0]
        labels = (y[:, None] == y[None, :]).float()
        mask = ~torch.eye(B, dtype=torch.bool, device=mu.device)
        # log-softmax over others
        logits = sims - 1e9 * (~mask)
        logprob = logits.log_softmax(dim=1)
        # positives: same y
        pos_logprob = (logprob * labels * mask).sum(dim=1) / (labels.sum(dim=1).clamp_min(1.0))
        return (-pos_logprob).to(mu.dtype)

    def _c_schedule(self, step: int, total_steps: int) -> float:
        # linear warmup to beta_max * z_dim / 2 (rough target KL)
        target = self.beta_max * (self.model.z_dim / 2.0)
        frac = min(1.0, step / max(total_steps, 1))
        return float(target * frac)

    def fit(self,
            X_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            t_train: Optional[np.ndarray] = None,
            y_train: Optional[np.ndarray] = None,
            epochs: int = 50,
            batch_size: int = 256,
            lr: float = 1e-3,
            patience: int = 8,
            outdir: Optional[Path] = None) -> "DVAETrainer":
        self.model.train()
        device = self.device
        X_tr = torch.from_numpy(np.asarray(X_train, dtype=np.float32)).to(device)
        T_tr = torch.from_numpy(t_train.astype(np.float32)).to(device) if t_train is not None else None
        Y_tr = torch.from_numpy(y_train.astype(np.int64)).to(device) if y_train is not None else None
        X_va = torch.from_numpy(np.asarray(X_val, dtype=np.float32)).to(device) if X_val is not None else None

        if self.model.in_dim is None:
            self.model.in_dim = int(X_tr.shape[1])
            self.model._init_modules()
            self.model.to(device)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        best = float("inf")
        best_state = None
        bad = 0

        total_steps = epochs * max(1, int(np.ceil(len(X_tr) / max(1, batch_size))))
        step = 0

        for ep in range(epochs):
            perm = torch.randperm(len(X_tr), device=device)
            ep_losses = {k: 0.0 for k in self.history.keys()}
            for i in range(0, len(X_tr), batch_size):
                idx = perm[i:i+batch_size]
                xb = X_tr[idx]
                tb = T_tr[idx] if T_tr is not None else None
                yb = Y_tr[idx] if Y_tr is not None else None

                x_tilde = self.corrupt(xb)
                x_hat, mu, logvar, _ = self.model(x_tilde, t=tb)

                # losses
                rec = F.l1_loss(x_hat, xb, reduction='none').mean(dim=1)
                kl = self._kl_normal(mu, logvar)
                c = self._c_schedule(step, total_steps)
                kl = torch.abs(kl - c)
                tcons = self._t_consistency(mu, yb)
                ctr = self._info_nce(mu, yb)
                spec = self._spec_loss(x_hat, xb)

                total = (self.w.rec * rec +
                         self.w.kl * kl +
                         self.w.tcons * tcons +
                         self.w.ctr * ctr +
                         self.w.spec * spec).mean()

                opt.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()

                for k, v in zip(["rec","kl","tcons","ctr","spec","total"], [rec,kl,tcons,ctr,spec,total.detach()]):
                    ep_losses[k] += float(v.mean().item())
                step += 1

            for k in ep_losses:
                ep_losses[k] /= max(1, int(np.ceil(len(X_tr) / max(1, batch_size))))
                self.history[k].append(ep_losses[k])

            # simple early stopping on validation reconstruction if available
            with torch.no_grad():
                if X_va is not None and len(X_va):
                    x_tilde = self.corrupt(X_va)
                    x_hat, _, _, _ = self.model(x_tilde)
                    val_rec = F.l1_loss(x_hat, X_va, reduction='none').mean().item()
                else:
                    val_rec = ep_losses["rec"]

            if outdir is not None:
                Path(outdir).mkdir(parents=True, exist_ok=True)
                with (Path(outdir) / "log.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": ep,
                                        **{f"train_{k}": v for k, v in ep_losses.items()},
                                        "val_rec": float(val_rec)}) + "\n")

            if val_rec < best - 1e-5:
                best = val_rec
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    @torch.no_grad()
    def transform(self, X: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
        self.model.eval()
        device = self.device
        x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        tb = None
        if self.model.use_t and (t is not None):
            tb = torch.from_numpy(t.astype(np.float32)).to(device)
        if self.model.enc is None:
            self.model.in_dim = int(x.shape[1])
            self.model._init_modules()
            self.model.to(device)
        # deterministic code: use mu from clean x
        h = self.model.enc(x)
        if self.model.use_t and tb is not None:
            h = self.model.film(h, fourier_embed_t(tb, dims=16))
        mu = self.model.mu(h)
        return mu.detach().cpu().numpy().astype(np.float32)

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state": self.model.state_dict(),
            "config": {
                "in_dim": self.model.in_dim,
                "z_dim": self.model.z_dim,
                "use_t": self.model.use_t,
                "hidden": self.model.hidden,
            }
        }, out)

    def load(self, path: str) -> "DVAETrainer":
        chk = torch.load(path, map_location=self.device)
        cfg = chk.get("config", {})
        self.model.in_dim = int(cfg.get("in_dim", self.model.in_dim or 0) or 0)
        self.model.z_dim = int(cfg.get("z_dim", self.model.z_dim))
        self.model.use_t = bool(cfg.get("use_t", self.model.use_t))
        self.model.hidden = int(cfg.get("hidden", self.model.hidden))
        if self.model.in_dim and self.model.enc is None:
            self.model._init_modules()
            self.model.to(self.device)
        self.model.load_state_dict(chk["state"])
        self.model.eval()
        return self
