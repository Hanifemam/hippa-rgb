"""Model definitions and simple factory for HIPPA RGB classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models

from models.late_fusion_embeddings import LateFusionHead


class Conv4DCNN(nn.Module):
    """
    Lightweight 4-layer CNN.

    Args:
        in_channels: number of input channels (e.g., 3 for RGB).
        num_classes: number of output classes.
        img_size: input image size (square).
        hidden_dim: hidden dimension for the first fully-connected layer.
        dropout: dropout probability before the output layer.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        img_size: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            x = self._forward_features(dummy)
            flat_dim = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = self.pool4(self.relu(self.conv4(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNet152Classifier(nn.Module):
    """ResNet152 backbone with optional dropout; all layers are trainable."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        img_size: int,
        hidden_dim: int,  # kept for signature compatibility
        dropout: float,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = tv_models.ResNet152_Weights.DEFAULT if pretrained else None
        self.model = tv_models.resnet152(weights=weights)

        # Unfreeze everything explicitly.
        for param in self.model.parameters():
            param.requires_grad = True

        if in_channels != self.model.conv1.in_channels:
            raise ValueError(f"ResNet152 expects {self.model.conv1.in_channels} input channels; got {in_channels}")

        in_features = self.model.fc.in_features
        layers = [
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.model.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet152LateFusion(nn.Module):
    """ResNet152 backbone feeding a LateFusionHead; all layers are trainable."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        img_size: int,
        hidden_dim: int,  # should be 512 for ResNet152 penultimate features
        dropout: float,
        num_cultivars: int | None = None,
        num_progressions: int | None = None,
        fusion_mode: str = "concat+sum+prod",
        pretrained: bool = True,
    ):
        super().__init__()
        weights = tv_models.ResNet152_Weights.DEFAULT if pretrained else None
        self.backbone = tv_models.resnet152(weights=weights)

        if in_channels != self.backbone.conv1.in_channels:
            raise ValueError(f"ResNet152 expects {self.backbone.conv1.in_channels} input channels; got {in_channels}")
        if hidden_dim != 512:
            raise ValueError(f"ResNet152LateFusion expects hidden_dim=512 (got {hidden_dim}) to match the penultimate layerwidth")

        for param in self.backbone.parameters():
            param.requires_grad = True

        # Remove the classifier; keep global pooling. Project to hidden_dim before fusion.
        feat_dim = self.backbone.fc.in_features if isinstance(self.backbone.fc, nn.Linear) else 2048
        self.backbone.fc = nn.Identity()
        proj_layers = [nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            proj_layers.append(nn.Dropout(p=dropout))
        self.proj = nn.Sequential(*proj_layers)
        self.head = LateFusionHead(
            feat_dim=hidden_dim,
            num_classes=num_classes,
            num_cultivars=num_cultivars,
            num_progressions=num_progressions,
            fusion_mode=fusion_mode,
            dropout=dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
        cultivar_ids: torch.Tensor | None = None,
        progression_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self.backbone(images)
        feats = self.proj(feats)
        return self.head(feats, cultivar_ids=cultivar_ids, progression_ids=progression_ids)


_MODEL_REGISTRY = {
    "conv4dcnn": Conv4DCNN,
    "resnet152": ResNet152Classifier,
    "resnet152_latefusion": ResNet152LateFusion,
}


class Conv4DCNNLateFusion(nn.Module):
    """
    Wrap Conv4DCNN and replace the final classifier with a LateFusionHead.
    Uses the Conv4DCNN layers to produce penultimate features (after dropout).
    """
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        img_size: int,
        hidden_dim: int,
        dropout: float,
        num_cultivars: int | None = None,
        num_progressions: int | None = None,
        fusion_mode: str = "concat+sum+prod",
    ):
        super().__init__()
        self.backbone = Conv4DCNN(
            in_channels=in_channels,
            num_classes=num_classes,  # placeholder; head handles classification
            img_size=img_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.head = LateFusionHead(
            feat_dim=hidden_dim,
            num_classes=num_classes,
            num_cultivars=num_cultivars,
            num_progressions=num_progressions,
            fusion_mode=fusion_mode,
            dropout=dropout,
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.pool1(self.backbone.relu(self.backbone.conv1(x)))
        x = self.backbone.pool2(self.backbone.relu(self.backbone.conv2(x)))
        x = self.backbone.pool3(self.backbone.relu(self.backbone.conv3(x)))
        x = self.backbone.pool4(self.backbone.relu(self.backbone.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.backbone.relu(self.backbone.fc1(x))
        return self.backbone.dropout(x)

    def forward(
        self,
        images: torch.Tensor,
        cultivar_ids: torch.Tensor | None = None,
        progression_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feats = self._features(images)
        return self.head(feats, cultivar_ids=cultivar_ids, progression_ids=progression_ids)


_MODEL_REGISTRY["conv4dcnn_latefusion"] = Conv4DCNNLateFusion


def build_model(model_name: str, **kwargs) -> nn.Module:
    """
    Create a model by name. Extend _MODEL_REGISTRY as new models are added.
    """
    name = model_name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)
