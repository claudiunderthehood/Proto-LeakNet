"""Proto-LeakNet: modular leak attribution and open-set evaluation.

This package factors the monolithic pipeline into importable modules for:
- data loading and splits
- feature construction on z_t latents
- lightweight encoder and attention
- prototype heads
- density backends (KDE/GMM) unified API
- calibration (per-class z-score)
- PCA sweep utilities
- compressors registry (PCA/DVAE)
- open-set decision and metrics
- a trainer that orchestrates training and evaluation

The design keeps density modeling (KDE/GMM) at the core and supports
calibrated OvR-LLR scoring and open-set rejection.
"""

try:  # Re-export while staying resilient to optional heavy deps.
    from .trainer import train_and_eval  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
    _IMPORT_ERROR = exc

    def train_and_eval(*args, **kwargs):  # type: ignore
        raise ModuleNotFoundError(
            "train_and_eval requires optional dependencies that are currently missing."
        ) from _IMPORT_ERROR

__all__ = [
    "train_and_eval",
]
