from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.late_fusion_embeddings import LateFusionHead

NUM_CLASSES = 9
IMG_SIZE = 224


# ============================================================
# Building blocks: CoordAtt, SpectralGate2D, SW-GAP, SAC
# ============================================================

class CoordAtt(nn.Module):
    """Coordinate Attention (used in improved ResNet)."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        x_h = self.pool_h(x)                    # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)        # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))

        out = identity * a_h * a_w
        return out


class SpectralGate2D(nn.Module):
    """
    Spectral gating G(x, ω):
    - FFT over spatial dims
    - magnitude → small MLP → gate in [0,1]
    - multiply with original feature map
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        freq = torch.fft.rfft2(x, dim=(-2, -1))                 # (B, C, H, W/2+1)
        mag = torch.abs(freq).mean(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
        y = F.relu(self.fc1(mag), inplace=True)
        gate = torch.sigmoid(self.fc2(y))                       # (B, C, 1, 1)
        return x * gate


class SWGAP(nn.Module):
    """
    Spectrally-Weighted Global Average Pooling (SW-GAP)
    X: (B, C, H, W)
    1) spectral energy per channel via FFT magnitude
    2) normalize energies => weights w_c
    3) GAP each channel
    4) output = w_c * GAP(channel)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1) spectral energy per channel
        freq = torch.fft.rfft2(x, dim=(-2, -1))           # (B, C, H, W/2+1)
        energy = torch.abs(freq).mean(dim=(-2, -1))       # (B, C)

        # 2) normalize channel weights
        w = energy / (energy.sum(dim=1, keepdim=True) + 1e-8)  # (B, C)

        # 3) GAP (spatial mean)
        gap = x.mean(dim=(-2, -1))                        # (B, C)

        # 4) apply spectral weights
        out = gap * w                                     # (B, C)
        return out


class SpectrallyAwareConcat(nn.Module):
    """
    Spectrally-Aware Concatenation (SAC):
    - Take branch vectors r, d, e (B, Cr/Cd/Ce)
    - Compute spectral descriptors via 1D FFT
    - Get per-branch weights with softmax
    - Scale each branch vector, then concatenate
    """

    def forward(self, r: torch.Tensor, d: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # r,d,e: (B, C_r/C_d/C_e)

        def branch_energy(v: torch.Tensor) -> torch.Tensor:
            freq = torch.fft.rfft(v, dim=1)           # (B, C/2+1)
            return torch.abs(freq).mean(dim=1, keepdim=True)  # (B, 1)

        er = branch_energy(r)
        ed = branch_energy(d)
        ee = branch_energy(e)

        energies = torch.cat([er, ed, ee], dim=1)         # (B, 3)
        weights = F.softmax(energies, dim=1)              # (B, 3)

        wr = weights[:, 0:1]
        wd = weights[:, 1:2]
        we = weights[:, 2:3]

        r_w = r * wr
        d_w = d * wd
        e_w = e * we

        return torch.cat([r_w, d_w, e_w], dim=1)          # (B, Cr+Cd+Ce)


# ============================================================
# Hybrid Backbone: ResNet50 + DenseNet121 + EfficientNetB0
# with improved ResNet, spectral gating, SW-GAP, and SAC
# ============================================================

class HMAFDD(nn.Module):
    """
    Hybrid Model for Apple Fruit Disease Detection (HMAFDD)

    - ResNet50 (improved with CoordAtt + multi-scale fusion)
    - DenseNet121 + spectral gating D(x)
    - EfficientNetB0 + spectral gating E(x)
    - SW-GAP for all branches
    - Spectrally-Aware Concatenation (SAC) for fusion
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.5,
        in_channels: int = 3,
        img_size: int = IMG_SIZE,
        hidden_dim: Optional[int] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"HMAFDD expects 3-channel RGB input; got {in_channels}")
        self.img_size = img_size
        self.hidden_dim = hidden_dim

        resnet_weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        densenet_weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        effnet_weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None

        # -------- ResNet50 (improved as in Zhang et al.) ----------
        resnet = models.resnet50(weights=resnet_weights)

        self.res_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.res_layer1 = resnet.layer1   # 256
        self.res_layer2 = resnet.layer2   # 512
        self.res_layer3 = resnet.layer3   # 1024
        self.res_layer4 = resnet.layer4   # 2048

        self.ca1 = CoordAtt(256)
        self.ca2 = CoordAtt(512)
        self.ca3 = CoordAtt(1024)
        self.ca4 = CoordAtt(2048)

        self.res_gap = SWGAP()
        self.res_proj1 = nn.Linear(256, 2048)
        self.res_proj2 = nn.Linear(512, 2048)
        self.res_proj3 = nn.Linear(1024, 2048)
        self.res_alpha = nn.Parameter(torch.ones(4))
        self.resnet_out_channels = 2048

        # -------- DenseNet121 + spectral gating ----------
        densenet = models.densenet121(weights=densenet_weights)
        self.densenet_features = densenet.features
        self.densenet_out_channels = densenet.classifier.in_features  # 1024
        self.dense_gate = SpectralGate2D(self.densenet_out_channels)
        self.dense_gap = SWGAP()

        # -------- EfficientNet-B0 + spectral gating ----------
        effnet = models.efficientnet_b0(
            weights=effnet_weights
        )
        self.effnet_features = effnet.features
        self.effnet_out_channels = effnet.classifier[1].in_features  # 1280
        self.eff_gate = SpectralGate2D(self.effnet_out_channels)
        self.eff_gap = SWGAP()

        # -------- Spectrally-Aware Concatenation (SAC) ----------
        self.sac = SpectrallyAwareConcat()

        # -------- Classifier Head ----------
        concat_dim = (
            self.resnet_out_channels
            + self.densenet_out_channels
            + self.effnet_out_channels
        )
        self.feature_dim = concat_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(concat_dim, num_classes)

    def _forward_resnet_branch(self, x):
        x = self.res_stem(x)

        x1 = self.ca1(self.res_layer1(x))   # (B, 256, ...)
        x2 = self.ca2(self.res_layer2(x1))  # (B, 512, ...)
        x3 = self.ca3(self.res_layer3(x2))  # (B, 1024, ...)
        x4 = self.ca4(self.res_layer4(x3))  # (B, 2048, ...)

        f1 = self.res_gap(x1)  # (B, 256)
        f2 = self.res_gap(x2)  # (B, 512)
        f3 = self.res_gap(x3)  # (B, 1024)
        f4 = self.res_gap(x4)  # (B, 2048)

        p1 = self.res_proj1(f1)
        p2 = self.res_proj2(f2)
        p3 = self.res_proj3(f3)
        p4 = f4

        w = torch.softmax(self.res_alpha, dim=0)  # (4,)
        fused = w[0] * p1 + w[1] * p2 + w[2] * p3 + w[3] * p4  # (B, 2048)
        return fused

    def _forward_densenet_branch(self, x):
        d = self.densenet_features(x)
        d = F.relu(d, inplace=True)
        d = self.dense_gate(d)
        d = self.dense_gap(d)                 # (B, 1024)
        return d

    def _forward_effnet_branch(self, x):
        e = self.effnet_features(x)
        e = self.eff_gate(e)
        e = self.eff_gap(e)                   # (B, 1280)
        return e

    def forward_backbones(self, x):
        r = self._forward_resnet_branch(x)
        d = self._forward_densenet_branch(x)
        e = self._forward_effnet_branch(x)
        return r, d, e

    def forward_features(self, x, apply_dropout: bool = True):
        r, d, e = self.forward_backbones(x)
        feat = self.sac(r, d, e)              # SAC instead of plain cat
        if apply_dropout:
            feat = self.dropout(feat)
        return feat

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.fc(feat)
        return logits


class HMAFDDLateFusion(nn.Module):
    """HMAFDD backbone paired with a LateFusionHead for cultivar/progression metadata."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        img_size: int,
        hidden_dim: Optional[int],
        dropout: float,
        num_cultivars: Optional[int] = None,
        num_progressions: Optional[int] = None,
        fusion_mode: str = "concat+sum+prod",
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = HMAFDD(
            num_classes=num_classes,
            dropout=dropout,
            in_channels=in_channels,
            img_size=img_size,
            hidden_dim=hidden_dim,
            pretrained=pretrained,
        )
        self.head = LateFusionHead(
            feat_dim=self.backbone.feature_dim,
            num_classes=num_classes,
            num_cultivars=num_cultivars,
            num_progressions=num_progressions,
            fusion_mode=fusion_mode,
            dropout=dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
        cultivar_ids: Optional[torch.Tensor] = None,
        progression_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = self.backbone.forward_features(images, apply_dropout=False)
        return self.head(feats, cultivar_ids=cultivar_ids, progression_ids=progression_ids)
