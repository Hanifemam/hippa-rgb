"""Model definitions and simple factory for HIPPA RGB classification."""

from __future__ import annotations

import torch
import torch.nn as nn


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


_MODEL_REGISTRY = {
    "conv4dcnn": Conv4DCNN,
}


def build_model(model_name: str, **kwargs) -> nn.Module:
    """
    Create a model by name. Extend _MODEL_REGISTRY as new models are added.
    """
    name = model_name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)
