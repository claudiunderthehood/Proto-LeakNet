from __future__ import annotations

from typing import Optional, Tuple

import torch


def load_sd21_vae(
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
    token: Optional[str] = None,
) -> Tuple[torch.nn.Module, float]:
    """Load SD2.1 VAE for latent-space encoding."""
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'diffusers' package is required for --latent-space true. "
            "Install it with `pip install diffusers accelerate`."
        ) from exc

    extra_kwargs = {"torch_dtype": dtype}
    if token:
        extra_kwargs["use_auth_token"] = token

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        subfolder="vae",
        **extra_kwargs,
    )
    vae.to(device, dtype=dtype)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    return vae, scaling


def encode_latents(
    vae: torch.nn.Module,
    images: torch.Tensor,
    *,
    scaling_factor: float,
) -> torch.Tensor:
    """Encode images ([-1,1] range) into SD2.1 latent space."""
    with torch.no_grad():
        posterior = vae.encode(images)
        latents = posterior.latent_dist.mean
    latents = latents * scaling_factor
    return latents

