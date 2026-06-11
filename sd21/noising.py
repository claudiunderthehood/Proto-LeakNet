from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor


ScheduleValue = Union[Tensor, Sequence[float]]


def get_noise_schedule(model: str = "sd21", num_steps: int = 1000) -> Dict[str, Tensor]:
    """Return alpha/sigma schedule compatible with Stable Diffusion 2.1.
    """
    m = model.lower()
    if m not in {"sd21", "stable-diffusion-2.1", "stable-diffusion-21"}:
        raise ValueError(f"Unsupported model '{model}'. Expected 'sd21'.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    device = torch.device("cpu")
    dtype = torch.float32
    beta_start, beta_end = 0.00085, 0.0120
    steps = torch.linspace(0, 1, num_steps, dtype=dtype, device=device)
    betas = beta_start + steps * (beta_end - beta_start)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_t = torch.sqrt(alpha_bar)
    sigma_t = torch.sqrt(1.0 - alpha_bar)

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "alpha_t": alpha_t,
        "sigma_t": sigma_t,
    }


def forward_noise(
    x0: Tensor,
    t: Union[int, Tensor, Sequence[int]],
    schedule: Dict[str, Tensor],
    *,
    generator: Optional[torch.Generator] = None,
    highpass: bool = False,
) -> Tensor:
    """Apply forward diffusion to produce z_t = alpha_t * x0 + sigma_t * eps."""
    if x0.ndim < 2:
        raise ValueError(f"x0 must be at least 2-D (B, ...), got shape {tuple(x0.shape)}")

    x = apply_highpass(x0) if highpass else x0
    return _forward_noise_core(x, t, schedule, generator=generator)


def forward_noise_latent(
    z0: Tensor,
    t: Union[int, Tensor, Sequence[int]],
    schedule: Dict[str, Tensor],
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Forward diffusion in latent space (no high-pass filtering)."""
    return _forward_noise_core(z0, t, schedule, generator=generator)


def normalize_by_sigma(
    z_t: Tensor,
    t: Union[int, Tensor, Sequence[int]],
    schedule: Dict[str, Tensor],
) -> Tensor:
    """Normalize noised samples by sigma_t (per sample)."""
    t_idx = _as_long_tensor(t, batch_size=z_t.shape[0], device=z_t.device)
    sigma = _gather_by_t(schedule.get("sigma_t"), t_idx, z_t)
    return z_t / (sigma + 1e-12)


def normalize_latent_by_sigma(
    z_t: Tensor,
    t: Union[int, Tensor, Sequence[int]],
    schedule: Dict[str, Tensor],
) -> Tensor:
    """Alias of normalize_by_sigma for latent tensors (clarity)."""
    return normalize_by_sigma(z_t, t, schedule)


def apply_highpass(x: Tensor) -> Tensor:
    """Laplacian high-pass filter preserving tensor dtype/device."""
    if x.ndim < 4:
        raise ValueError("High-pass filter expects (B, C, H, W) tensor.")
    kernel = _laplacian_kernel(device=x.device, dtype=x.dtype, channels=x.shape[1])
    padding = 1
    return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])


def _laplacian_kernel(device: torch.device, dtype: torch.dtype, channels: int) -> Tensor:
    base = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
    kernel = base.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


def _as_long_tensor(
    t: Union[int, Tensor, Sequence[int]],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if isinstance(t, Tensor):
        t_long = t.to(device=device, dtype=torch.long)
    elif isinstance(t, Iterable):
        t_long = torch.tensor(list(int(v) for v in t), device=device, dtype=torch.long)
    else:
        t_long = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
    if t_long.ndim == 0:
        t_long = t_long.expand(batch_size)
    if t_long.shape[0] != batch_size:
        if t_long.numel() == 1:
            t_long = t_long.expand(batch_size)
        else:
            raise ValueError(f"Mismatch between t ({t_long.shape[0]}) and batch size ({batch_size}).")
    return t_long


def _gather_by_t(value: Optional[ScheduleValue], t_idx: Tensor, reference: Tensor) -> Tensor:
    if value is None:
        raise ValueError("Schedule is missing required entries (alpha_t/sigma_t).")
    if not isinstance(value, Tensor):
        value = torch.tensor(list(value), device=reference.device, dtype=reference.dtype)
    value = value.to(device=reference.device, dtype=reference.dtype)
    gathered = value.index_select(0, t_idx.view(-1))
    new_shape = (t_idx.shape[0],) + (1,) * (reference.ndim - 1)
    return gathered.view(new_shape)


def _forward_noise_core(
    x0: Tensor,
    t: Union[int, Tensor, Sequence[int]],
    schedule: Dict[str, Tensor],
    *,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if x0.ndim < 2:
        raise ValueError(f"x0 must be at least 2-D (B, ...), got shape {tuple(x0.shape)}")
    t_idx = _as_long_tensor(t, batch_size=x0.shape[0], device=x0.device)
    alpha = _gather_by_t(schedule.get("alpha_t"), t_idx, x0)
    sigma = _gather_by_t(schedule.get("sigma_t"), t_idx, x0)
    if generator is not None:
        eps = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
    else:
        eps = torch.randn_like(x0)
    return alpha * x0 + sigma * eps
