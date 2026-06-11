from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models as _tv_models
    try:
        from torchvision.models import ResNet18_Weights as _TV_RESNET18_WEIGHTS
    except ImportError:  # torchvision<0.13 fallback
        _TV_RESNET18_WEIGHTS = None
    try:
        from torchvision.models import ResNet50_Weights as _TV_RESNET50_WEIGHTS
    except ImportError:
        _TV_RESNET50_WEIGHTS = None
    try:
        from torchvision.models import ResNet101_Weights as _TV_RESNET101_WEIGHTS
    except ImportError:
        _TV_RESNET101_WEIGHTS = None
    try:
        from torchvision.models import EfficientNet_B4_Weights as _TV_EFFNET_B4_WEIGHTS
    except Exception:
        _TV_EFFNET_B4_WEIGHTS = None
    try:
        from torchvision.models import ViT_B_16_Weights as _TV_VIT_B16_WEIGHTS
    except Exception:
        _TV_VIT_B16_WEIGHTS = None
except Exception:  # torchvision optional dependency
    _tv_models = None
    _TV_RESNET18_WEIGHTS = None
    _TV_RESNET50_WEIGHTS = None
    _TV_RESNET101_WEIGHTS = None
    _TV_EFFNET_B4_WEIGHTS = None
    _TV_VIT_B16_WEIGHTS = None


class LeakEncoder(nn.Module):
    """Lightweight encoder for leak-centric inputs.

    A small ConvNet with global pooling producing an embedding vector.
    """

    def __init__(self, in_ch: int, embed_dim: int = 96, anti_semantic: bool = True):
        super().__init__()
        self.anti_semantic = anti_semantic
        ch = [in_ch, 64, 128, 192]
        self.conv1 = nn.Conv2d(ch[0], ch[1], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch[1], ch[2], kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(ch[1])
        self.bn2 = nn.BatchNorm2d(ch[2])
        self.bn3 = nn.BatchNorm2d(ch[3])
        self.head = nn.Linear(ch[3], embed_dim)
        self.dropout = nn.Dropout(p=0.2 if anti_semantic else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        h = self.head(x)
        return h


class ResNetLeakEncoder(nn.Module):
    """ResNet encoder (18/50/101) that adapts to arbitrary channel counts."""

    def __init__(
        self,
        in_ch: int,
        embed_dim: int = 128,
        *,
        anti_semantic: bool = True,
        variant: str = "resnet18",
        pretrained: bool = True,
        freeze_stem: bool = False,
    ) -> None:
        super().__init__()
        if _tv_models is None:
            raise RuntimeError("torchvision is required for the ResNet backbone (install torchvision).")

        v = (variant or "resnet18").lower()
        if v in {"resnet", "resnet18", "18"}:
            constructor = _tv_models.resnet18
            weights_enum = _TV_RESNET18_WEIGHTS
        elif v in {"resnet50", "50"}:
            constructor = _tv_models.resnet50
            weights_enum = _TV_RESNET50_WEIGHTS
        elif v in {"resnet101", "101"}:
            constructor = _tv_models.resnet101
            weights_enum = _TV_RESNET101_WEIGHTS
        else:
            raise ValueError(f"Unsupported ResNet variant '{variant}'. Expected resnet18, resnet50, or resnet101.")

        if weights_enum is not None:
            weights = weights_enum.DEFAULT if pretrained else None
            base = constructor(weights=weights)
        else:
            base = constructor(pretrained=pretrained)

        if in_ch != 3:
            conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                w = base.conv1.weight
                if pretrained:
                    if in_ch > 3:
                        conv1.weight[:, :3].copy_(w)
                        mean_w = w.mean(dim=1, keepdim=True)
                        conv1.weight[:, 3:].copy_(mean_w.repeat(1, in_ch - 3, 1, 1))
                    else:  # in_ch < 3
                        take = min(in_ch, w.shape[1])
                        if take > 0:
                            conv1.weight[:, :take].copy_(w[:, :take])
                else:
                    nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
            base.conv1 = conv1

        if freeze_stem:
            for name, param in base.named_parameters():
                if name.startswith('conv1') or name.startswith('bn1'):
                    param.requires_grad = False

        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.head = nn.Linear(in_features, embed_dim)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.dropout = nn.Dropout(p=0.2 if anti_semantic else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base(x)
        return self.head(h)


class EfficientNetLeakEncoder(nn.Module):
    """EfficientNet-B4 based encoder that adapts to arbitrary channel counts."""

    def __init__(
        self,
        in_ch: int,
        embed_dim: int = 128,
        *,
        anti_semantic: bool = True,
        variant: str = "b4",
        pretrained: bool = True,
        freeze_stem: bool = False,
    ) -> None:
        super().__init__()
        if _tv_models is None:
            raise RuntimeError("torchvision is required for the EfficientNet backbone (install torchvision).")

        v = (variant or "b4").lower()
        if v not in {"b4"}:
            raise ValueError(f"Only EfficientNet b4 supported currently, got variant={variant!r}")

        if _TV_EFFNET_B4_WEIGHTS is not None:
            weights = _TV_EFFNET_B4_WEIGHTS.DEFAULT if pretrained else None
            base = _tv_models.efficientnet_b4(weights=weights)
        else:
            base = _tv_models.efficientnet_b4(pretrained=pretrained)

        # Adapt first conv to arbitrary input channels
        stem = base.features[0][0]
        if in_ch != 3:
            conv1 = nn.Conv2d(in_ch, stem.out_channels, kernel_size=stem.kernel_size, stride=stem.stride,
                              padding=stem.padding, bias=False)
            with torch.no_grad():
                w = stem.weight
                if pretrained:
                    if in_ch > 3:
                        conv1.weight[:, :3].copy_(w)
                        mean_w = w.mean(dim=1, keepdim=True)
                        conv1.weight[:, 3:].copy_(mean_w.repeat(1, in_ch - 3, 1, 1))
                    else:  # in_ch < 3
                        take = min(in_ch, w.shape[1])
                        if take > 0:
                            conv1.weight[:, :take].copy_(w[:, :take])
                else:
                    nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
            base.features[0][0] = conv1

        if freeze_stem:
            for p in base.features[0].parameters():
                p.requires_grad = False

        # Grab classifier input dim then drop classifier and keep pooling
        in_features = None
        if isinstance(base.classifier, nn.Linear):
            in_features = base.classifier.in_features
        elif isinstance(base.classifier, nn.Sequential):
            for m in reversed(base.classifier):
                if isinstance(m, nn.Linear):
                    in_features = m.in_features
                    break
        if in_features is None:
            in_features = 1792  # fallback for b4

        # Replace classifier with identity to output pooled feature vector
        base.classifier = nn.Identity()
        self.base = base
        self.head = nn.Linear(in_features, embed_dim)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.dropout = nn.Dropout(p=0.2 if anti_semantic else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.base(x)  # returns pooled flattened features because classifier is Identity
        return self.head(h)


class ViTLeakEncoder(nn.Module):
    """Vision Transformer encoder that adapts the patch embed to arbitrary channels."""

    def __init__(
        self,
        in_ch: int,
        embed_dim: int = 128,
        *,
        anti_semantic: bool = True,
        variant: str = "b16",
        pretrained: bool = True,
        freeze_patch_embed: bool = False,
    ) -> None:
        super().__init__()
        if _tv_models is None:
            raise RuntimeError("torchvision is required for the ViT backbone (install torchvision).")

        v = (variant or "b16").lower()
        if v in {"b16", "vit_b_16", "vit-b16"}:
            if not hasattr(_tv_models, "vit_b_16"):
                raise RuntimeError("Current torchvision build does not provide vit_b_16.")
            if _TV_VIT_B16_WEIGHTS is not None:
                weights = _TV_VIT_B16_WEIGHTS.DEFAULT if pretrained else None
                base = _tv_models.vit_b_16(weights=weights)
            else:
                base = _tv_models.vit_b_16(pretrained=pretrained)
        else:
            raise ValueError(f"Only ViT b16 supported currently, got variant={variant!r}")

        patch_conv = base.conv_proj
        if in_ch != patch_conv.in_channels:
            conv_proj = nn.Conv2d(
                in_ch,
                patch_conv.out_channels,
                kernel_size=patch_conv.kernel_size,
                stride=patch_conv.stride,
                bias=False,
            )
            with torch.no_grad():
                w = patch_conv.weight
                if pretrained:
                    if in_ch > w.shape[1]:
                        conv_proj.weight[:, : w.shape[1]].copy_(w)
                        mean_w = w.mean(dim=1, keepdim=True)
                        conv_proj.weight[:, w.shape[1] :].copy_(mean_w.repeat(1, in_ch - w.shape[1], 1, 1))
                    else:
                        take = min(in_ch, w.shape[1])
                        if take > 0:
                            conv_proj.weight[:, :take].copy_(w[:, :take])
                else:
                    nn.init.kaiming_normal_(conv_proj.weight, mode='fan_out', nonlinearity='relu')
            base.conv_proj = conv_proj

        if freeze_patch_embed:
            for p in base.conv_proj.parameters():
                p.requires_grad = False

        hidden_dim = getattr(base, "hidden_dim", None)
        if hidden_dim is None:
            raise RuntimeError("Unable to determine ViT hidden dimension.")

        base.heads = nn.Identity()
        self.required_image_size = getattr(base, "image_size", None)
        self.base = base
        self.head = nn.Linear(hidden_dim, embed_dim)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.dropout = nn.Dropout(p=0.2 if anti_semantic else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.required_image_size is not None:
            h, w = x.shape[-2:]
            if h != self.required_image_size or w != self.required_image_size:
                x = F.interpolate(x, size=(self.required_image_size, self.required_image_size), mode="bilinear", align_corners=False)
        h = self.base(x)
        return self.head(h)


class FeatureAttention(nn.Module):
    """Simple channel-wise attention over the embedding for weighted distances."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim), nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.gate(h)


class LeakBackbone(nn.Module):
    """Wrapper combining encoder and an optional attention head."""

    def __init__(
        self,
        in_ch: int,
        embed_dim: int = 96,
        use_attention: bool = True,
        anti_semantic: bool = True,
        encoder_type: str = "conv",
        *,
        resnet_variant: str = "resnet18",
        resnet_pretrained: bool = True,
        resnet_freeze_stem: bool = False,
        effnet_variant: str = "b4",
        effnet_pretrained: bool = True,
        effnet_freeze_stem: bool = False,
        vit_variant: str = "b16",
        vit_pretrained: bool = True,
        vit_freeze_patch: bool = False,
    ) -> None:
        super().__init__()
        etype = encoder_type.lower()
        if etype in {"conv", "convnet", "cnn"}:
            self.encoder = LeakEncoder(in_ch=in_ch, embed_dim=embed_dim, anti_semantic=anti_semantic)
        elif etype in {"resnet", "resnet18", "resnet-18", "resnet50", "resnet-50", "resnet101", "resnet-101"}:
            variant_alias = {
                "resnet": resnet_variant,
                "resnet18": "resnet18",
                "resnet-18": "resnet18",
                "resnet50": "resnet50",
                "resnet-50": "resnet50",
                "resnet101": "resnet101",
                "resnet-101": "resnet101",
            }
            resolved_variant = variant_alias.get(etype, resnet_variant) or "resnet18"
            self.encoder = ResNetLeakEncoder(
                in_ch=in_ch,
                embed_dim=embed_dim,
                anti_semantic=anti_semantic,
                variant=resolved_variant,
                pretrained=resnet_pretrained,
                freeze_stem=resnet_freeze_stem,
            )
        elif etype in {"efficientnet", "effnet", "efficientnet_b4"}:
            self.encoder = EfficientNetLeakEncoder(
                in_ch=in_ch,
                embed_dim=embed_dim,
                anti_semantic=anti_semantic,
                variant=effnet_variant,
                pretrained=effnet_pretrained,
                freeze_stem=effnet_freeze_stem,
            )
        elif etype in {"vit", "visiontransformer", "vit_b16", "vit-b16"}:
            self.encoder = ViTLeakEncoder(
                in_ch=in_ch,
                embed_dim=embed_dim,
                anti_semantic=anti_semantic,
                variant=vit_variant,
                pretrained=vit_pretrained,
                freeze_patch_embed=vit_freeze_patch,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type '{encoder_type}'. Expected 'conv', 'resnet18', 'resnet50', 'resnet101', 'efficientnet', or 'vit'."
            )
        self.att = FeatureAttention(embed_dim) if use_attention else None
        self.encoder_type = etype

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.encoder(x)
        w = self.att(h) if self.att is not None else None
        return h, w


class OpenHead(nn.Module):
    """Binary open-vs-closed head on top of embeddings.

    Uses a small MLP with a single hidden layer. Output is a single logit.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h).squeeze(1)
