from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

try:
    from torchvision.models import ResNet18_Weights, EfficientNet_B4_Weights
except Exception:  # torchvision optional
    ResNet18_Weights = None
    EfficientNet_B4_Weights = None


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    class_name: str


class ClosedSetDataset(Dataset):
    """Dataset returning (image_tensor, label, path)."""

    def __init__(self, samples: Sequence[Sample], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        img = _load_image(sample.path)
        tensor = self.transform(img)
        return tensor, sample.label, str(sample.path)


class OpenSetDataset(Dataset):
    """Open-set dataset returning label=-1."""

    def __init__(self, paths: Sequence[Path], transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        img = _load_image(path)
        tensor = self.transform(img)
        return tensor, -1, str(path)


@dataclass
class DatasetBundle:
    class_to_idx: Dict[str, int]
    train: ClosedSetDataset
    val: ClosedSetDataset
    test: ClosedSetDataset
    open_set: Optional[OpenSetDataset]
    transform_meta: Dict[str, str]
    golden_batch: Dict[str, List[str]]
    channel_stats: Optional[Tuple[torch.Tensor, torch.Tensor]]


def build_closed_open_datasets(
    closed_root: Path,
    open_root: Optional[Path],
    *,
    backbone: str = "conv",
    pretrained: bool = True,
    image_size: int = 256,
    compute_channel_stats: bool = False,
    latent_mode: bool = False,
    seed: int = 0,
) -> DatasetBundle:
    """Discover closed/open splits and attach transforms without double normalization."""
    closed_root = Path(closed_root)
    open_root = Path(open_root) if open_root is not None else None
    if not closed_root.exists():
        raise FileNotFoundError(f"Closed set root '{closed_root}' not found.")

    class_dirs = sorted([p for p in closed_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {closed_root}.")

    class_to_idx = {cls.name: idx for idx, cls in enumerate(class_dirs)}
    split = _resolve_splits(closed_root, class_to_idx)

    rng = random.Random(seed)
    train_samples = _build_samples(split["train"], class_to_idx, root=closed_root, rng=rng)
    val_samples = _build_samples(split["val"], class_to_idx, root=closed_root, rng=None)
    test_samples = _build_samples(split["test"], class_to_idx, root=closed_root, rng=None)

    use_weights, transform_meta = _should_use_weights(backbone, pretrained)
    channel_stats = None

    if latent_mode:
        train_transform, eval_transform = _latent_transforms(image_size)
        transform_meta["transform_name"] = "SD21-VAE"
        transform_meta["latent_space"] = "true"
        channel_stats = None
        use_weights = False
    elif use_weights:
        train_transform = transform_meta["weights"].transforms()
        eval_transform = transform_meta["weights"].transforms()
    else:
        train_transform, eval_transform = _default_transforms(image_size, normalize_mean_std=None)
        if compute_channel_stats:
            mean, std = _compute_channel_stats(train_samples, image_size)
            channel_stats = (mean, std)
            train_transform, eval_transform = _default_transforms(image_size, normalize_mean_std=(mean, std))
        else:
            mean = torch.full((3,), 0.5)
            std = torch.full((3,), 0.5)
            channel_stats = (mean, std)
            train_transform, eval_transform = _default_transforms(image_size, normalize_mean_std=(mean, std))

    bundle = DatasetBundle(
        class_to_idx=class_to_idx,
        train=ClosedSetDataset(train_samples, train_transform),
        val=ClosedSetDataset(val_samples, eval_transform),
        test=ClosedSetDataset(test_samples, eval_transform),
        open_set=_build_open_dataset(open_root, eval_transform),
        transform_meta=_summarize_transforms(transform_meta, image_size, use_weights, latent_mode),
        golden_batch=_make_golden_batch(
            {"train": train_samples, "val": val_samples, "test": test_samples}, per_class=1, limit=24
        ),
        channel_stats=channel_stats if not use_weights and not latent_mode else None,
    )
    return bundle


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.copy()


def _resolve_splits(closed_root: Path, class_to_idx: Dict[str, int]):
    split_path = closed_root / "Split.json"
    if split_path.exists():
        return _parse_split_json(split_path, closed_root, class_to_idx)

    split = {"train": {}, "val": {}, "test": {}}
    for cls_name in class_to_idx.keys():
        files = sorted(_list_images(closed_root / cls_name))
        n = len(files)
        if n == 0:
            continue
        idx_train = max(1, int(round(0.7 * n)))
        idx_val = max(idx_train + 1, int(round(0.85 * n)))
        idx_val = min(idx_val, n)
        idx_train = min(idx_train, n)
        train_files = files[:idx_train]
        val_files = files[idx_train:idx_val]
        test_files = files[idx_val:]
        if not val_files and test_files:
            val_files, test_files = test_files[:1], test_files[1:]
        if not test_files and val_files:
            test_files, val_files = val_files[-1:], val_files[:-1]
        split["train"][cls_name] = [f.name for f in train_files]
        split["val"][cls_name] = [f.name for f in val_files]
        split["test"][cls_name] = [f.name for f in test_files]
    return split


def _parse_split_json(split_path: Path, closed_root: Path, class_to_idx: Dict[str, int]):
    with split_path.open("r") as f:
        raw = json.load(f)
    split: Dict[str, Dict[str, List[str]]] = {"train": {}, "val": {}, "test": {}}

    key_map = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }

    class_files: Dict[str, List[Path]] = {
        cls_name: sorted(_list_images(closed_root / cls_name), key=lambda p: p.name.lower())
        for cls_name in class_to_idx
    }

    for raw_key, value in raw.items():
        canonical = key_map.get(raw_key.lower())
        if canonical is None:
            continue
        if isinstance(value, dict):
            for cls_name in class_to_idx:
                entries = value.get(cls_name, [])
                split[canonical][cls_name] = [str(item) for item in entries]
        elif isinstance(value, list):
            indices = [int(x) for x in value]
            for cls_name in class_to_idx:
                files = class_files.get(cls_name, [])
                selected: List[str] = []
                for idx in indices:
                    if idx < 0 or idx >= len(files):
                        raise IndexError(
                            f"Split index {idx} out of bounds for class '{cls_name}' (found {len(files)} files)."
                        )
                    selected.append(files[idx].name)
                split[canonical][cls_name] = selected
        else:
            raise ValueError(f"Unsupported Split.json format for key '{raw_key}'.")

    return split


def _build_samples(
    split_dict: Dict[str, List[str]],
    class_to_idx: Dict[str, int],
    *,
    root: Path,
    rng: Optional[random.Random],
) -> List[Sample]:
    samples: List[Sample] = []
    for cls_name, filenames in split_dict.items():
        cls_idx = class_to_idx[cls_name]
        paths = [root / cls_name / fname for fname in filenames]
        samples.extend(Sample(path=path, label=cls_idx, class_name=cls_name) for path in paths)
    samples.sort(key=lambda s: (s.class_name, s.path.name))
    if rng is not None:
        rng.shuffle(samples)
    return samples


def _build_open_dataset(open_root: Optional[Path], transform):
    if open_root is None or not open_root.exists():
        return None
    paths = sorted(_list_images(open_root))
    return OpenSetDataset(paths, transform)


def _list_images(root: Path) -> List[Path]:
    if root.is_dir():
        return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if root.is_file():
        return [root]
    return []


def _should_use_weights(backbone: str, pretrained: bool):
    backbone = (backbone or "conv").lower()
    meta: Dict[str, object] = {"backbone": backbone, "pretrained": str(pretrained)}
    if backbone in {"resnet18", "resnet"} and pretrained and ResNet18_Weights is not None:
        weights = ResNet18_Weights.DEFAULT
        meta["weights"] = weights
        meta["transform_name"] = "ResNet18_Weights.DEFAULT"
        return True, meta
    if backbone in {"efficientnet_b4", "efficientnet", "effnet"} and pretrained and EfficientNet_B4_Weights is not None:
        weights = EfficientNet_B4_Weights.DEFAULT
        meta["weights"] = weights
        meta["transform_name"] = "EfficientNet_B4_Weights.DEFAULT"
        return True, meta
    meta["transform_name"] = "LeakNetDefault"
    return False, meta


def _default_transforms(image_size: int, normalize_mean_std: Optional[Tuple[torch.Tensor, torch.Tensor]]):
    resize = T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    center = T.CenterCrop(image_size)
    to_tensor = T.ToTensor()
    if normalize_mean_std is None:
        return T.Compose([resize, center, to_tensor]), T.Compose([resize, center, to_tensor])
    mean, std = normalize_mean_std
    norm = T.Normalize(mean.tolist(), std.tolist())
    base = [resize, center, to_tensor, norm]
    return T.Compose(base), T.Compose(base)


def _latent_transforms(image_size: int):
    resize = T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    center = T.CenterCrop(image_size)
    to_tensor = T.ToTensor()
    norm = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    base = [resize, center, to_tensor, norm]
    comp = T.Compose(base)
    return comp, comp


def _compute_channel_stats(samples: Sequence[Sample], image_size: int, max_items: int = 4096):
    if not samples:
        raise RuntimeError("Cannot compute channel stats without samples.")
    resize = T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)
    center = T.CenterCrop(image_size)
    to_tensor = T.ToTensor()
    tensors: List[torch.Tensor] = []
    for sample in samples[:max_items]:
        img = _load_image(sample.path.parent / sample.path.name if sample.path.is_absolute() else sample.path)
        tensors.append(to_tensor(center(resize(img))))
    stacked = torch.stack(tensors, dim=0)
    mean = stacked.mean(dim=[0, 2, 3])
    std = stacked.std(dim=[0, 2, 3])
    std = torch.clamp(std, min=1e-3)
    return mean, std


def _make_golden_batch(
    splits: Dict[str, Sequence[Sample]],
    *,
    per_class: int,
    limit: int,
) -> Dict[str, List[str]]:
    golden: Dict[str, List[str]] = {}
    for split_name, samples in splits.items():
        bucket: Dict[str, List[str]] = {}
        ordered_samples = sorted(samples, key=lambda s: (s.class_name, s.path.name))
        for sample in ordered_samples:
            bucket.setdefault(sample.class_name, [])
            if len(bucket[sample.class_name]) < per_class:
                bucket[sample.class_name].append(str(sample.path))
        # Flatten preserving class order
        ordered = []
        for class_name in sorted(bucket.keys()):
            ordered.extend(bucket[class_name])
        golden[split_name] = ordered[:limit]
    return golden


def _summarize_transforms(
    meta: Dict[str, object],
    image_size: int,
    used_pretrained: bool,
    latent_mode: bool,
) -> Dict[str, str]:
    summary = {
        "backbone": str(meta.get("backbone", "")),
        "pretrained": str(meta.get("pretrained", "")),
        "transform_name": str(meta.get("transform_name", "")),
        "image_size": str(image_size),
        "double_normalize": "false",
        "input_space": "latent" if latent_mode else "image",
    }
    if latent_mode:
        summary["note"] = "SD21 latent prep (Resize/CenterCrop/ToTensor/Normalize[0.5,0.5])"
    elif used_pretrained:
        summary["note"] = "using torchvision weights transforms()"
    else:
        summary["note"] = "Resize/CenterCrop/ToTensor/Normalize"
    return summary
