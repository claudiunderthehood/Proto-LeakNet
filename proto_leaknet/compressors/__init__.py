from __future__ import annotations

"""Compressors registry for Proto-LeakNet.

Currently supported:
- none: no compression (identity)
- pca: handled in trainer (legacy path)
- dvae: denoising VAE on embedding vectors

The compressors expose a common minimal API:
- fit(X_train, X_val, t_train=None, y_train=None): optional
- transform(X, t=None): returns compressed features
- save(path), load(path): serialization helpers
"""

from typing import Optional


def build_compressor(name: str,
                     latent_dim: int = 128,
                     use_t: bool = False,
                     device: str = "cpu"):
    name = (name or "none").lower()
    if name in {"none", "identity"}:
        return IdentityCompressor()
    if name == "dvae":
        from .dvae import LeakDVAE, DVAETrainer
        model = LeakDVAE(in_dim=None,  # set on first call
                         z_dim=int(latent_dim),
                         use_t=bool(use_t)).to_device(device)
        trainer = DVAETrainer(model=model, device=device)
        return trainer
    if name == "pca":
        # handled in trainer directly (kept for CLI symmetry)
        return IdentityCompressor()
    raise ValueError(f"Unknown compressor: {name}")


class IdentityCompressor:
    def fit(self, X_train, X_val=None, t_train=None, y_train=None):
        return self

    def transform(self, X, t=None):
        return X

    def save(self, path: str):
        return None

    def load(self, path: str):
        return self

