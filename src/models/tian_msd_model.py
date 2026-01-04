"""
Compact Tian et al. (2021) Multi-scale Dense Inception models (model-only).

Faithful mechanisms:
- H_i transform: BN → ReLU → Conv(1×1) → BN → ReLU → Conv(3×3)
- Dense cascading over repeated Inception blocks within each stage
- Multi-scale connection: convolved Stem output spliced with Dense stage outputs before pooling

Practical adaptation:
- Uses timm pretrained backbones; adds 1×1 adapters to keep channel sizes compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class HiTransform(nn.Module):
    """BN → ReLU → Conv(1×1) → BN → ReLU → Conv(3×3)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv1(x)
        x = F.relu(self.bn2(x), inplace=True)
        return self.conv3(x)


class DenseRepeatStage(nn.Module):
    """
    Dense cascading over repeated Inception blocks.

    For each block:
    - apply block
    - apply H_i to block output
    - concatenate all H outputs so far (dense connectivity)
    - adapt concatenation back to the expected channel size for the next block
      (needed to remain compatible with pretrained timm blocks).
    """

    def __init__(self, blocks: nn.ModuleList, growth: int = 128):
        super().__init__()
        self.blocks = blocks
        self.growth = growth
        self.hi = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self._built = False
        self.out_ch: int | None = None
        self.last_cur: torch.Tensor | None = None

    @torch.no_grad()
    def _build(self, x: torch.Tensor) -> None:
        device = x.device
        prev: list[torch.Tensor] = []
        cur = x

        for blk in self.blocks:
            y = blk(cur)
            hi = HiTransform(y.shape[1], self.growth).to(device)
            self.hi.append(hi)

            dense_ch = (len(prev) + 1) * self.growth
            adapt = nn.Sequential(
                nn.Conv2d(dense_ch, y.shape[1], 1, bias=False),
                nn.BatchNorm2d(y.shape[1]),
                nn.ReLU(inplace=True),
            ).to(device)
            self.adapters.append(adapt)

            z = hi(y)
            prev.append(z)
            cur = adapt(torch.cat(prev, dim=1))

        self.out_ch = len(prev) * self.growth
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x)

        prev: list[torch.Tensor] = []
        cur = x
        for hi, adapt, blk in zip(self.hi, self.adapters, self.blocks):
            y = blk(cur)
            z = hi(y)
            prev.append(z)
            cur = adapt(torch.cat(prev, dim=1))

        self.last_cur = cur
        return torch.cat(prev, dim=1)


class MultiScaleHead(nn.Module):
    """
    Multi-scale connection head:
    - Convolve stem output (paper: stem is convolved before splicing)
    - Optionally convolve each stage output (light projection) before splicing
    - Resize stage outputs to stem spatial size
    - Concatenate and apply GAP + FC
    """

    def __init__(
        self,
        stem_ch: int,
        stage_chs: list[int],
        num_classes: int,
        dropout: float = 0.8,
        proj_stem: bool = True,
        proj_stages: bool = True,
    ):
        super().__init__()

        self.stem_proj = (
            nn.Sequential(
                nn.Conv2d(stem_ch, stem_ch, 1, bias=False),
                nn.BatchNorm2d(stem_ch),
                nn.ReLU(inplace=True),
            )
            if proj_stem
            else nn.Identity()
        )

        self.stage_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ) if proj_stages else nn.Identity()
            for ch in stage_chs
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(stem_ch + sum(stage_chs), num_classes)

    def _fuse(self, stem: torch.Tensor, stages: list[torch.Tensor]) -> torch.Tensor:
        stem = self.stem_proj(stem)
        H, W = stem.shape[-2:]
        feats = [stem]

        for proj, s in zip(self.stage_proj, stages):
            s = proj(s)
            s = F.interpolate(s, size=(H, W), mode="bilinear", align_corners=False)
            feats.append(s)

        x = torch.cat(feats, dim=1)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, stem: torch.Tensor, stages: list[torch.Tensor], return_feats: bool = False):
        fused = self._fuse(stem, stages)
        feats = self.dropout(fused)
        logits = self.fc(feats)
        if return_feats:
            return logits, feats
        return logits


# ------------------------------------------------------------
# Backbone decomposition helpers (timm) — robust across versions
# ------------------------------------------------------------
def _split_inception_v4(model):
    # Most timm versions expose `features` as a Sequential
    if hasattr(model, "features"):
        feats = list(model.features.children())
        if len(feats) < 6:
            raise ValueError("Unexpected InceptionV4 'features' layout.")
        stem = feats[0]
        A, redA, B, redB, C = feats[1:6]
        return stem, list(A.children()), redA, list(B.children()), redB, list(C.children())

    # Some versions expose stem + named groups
    if hasattr(model, "stem"):
        if hasattr(model, "inception_a") and hasattr(model, "inception_b") and hasattr(model, "inception_c"):
            stem = model.stem
            A = list(getattr(model, "inception_a", nn.Sequential()).children())
            redA = getattr(model, "reduction_a", None)
            B = list(getattr(model, "inception_b", nn.Sequential()).children())
            redB = getattr(model, "reduction_b", None)
            C = list(getattr(model, "inception_c", nn.Sequential()).children())
            if redA is None or redB is None:
                raise ValueError("Could not locate reduction blocks in InceptionV4.")
            return stem, A, redA, B, redB, C

    # Fallback: stem + flat blocks list
    stem = getattr(model, "stem", None)
    blocks = list(getattr(model, "blocks", []))
    if stem is None or not blocks:
        raise AttributeError("Unexpected InceptionV4 structure; missing 'features'/'stem'/'blocks'.")

    A: list[nn.Module] = []
    B: list[nn.Module] = []
    C: list[nn.Module] = []
    redA = redB = None
    current = A

    for blk in blocks:
        name = blk.__class__.__name__.lower()
        if "reductiona" in name or "reduction_a" in name:
            redA = blk
            current = B
            continue
        if "reductionb" in name or "reduction_b" in name:
            redB = blk
            current = C
            continue
        current.append(blk)

    if redA is None or redB is None:
        raise ValueError("Could not locate reduction blocks in InceptionV4 fallback split.")
    return stem, A, redA, B, redB, C


def _split_inception_resnet_v2(model):
    # Common timm layout: features Sequential
    if hasattr(model, "features"):
        feats = list(model.features.children())
        if len(feats) < 6:
            raise ValueError("Unexpected InceptionResNetV2 'features' layout.")
        stem = feats[0]
        A, redA, B, redB, C = feats[1:6]
        return stem, list(A.children()), redA, list(B.children()), redB, list(C.children())

    # timm exposes explicit repeat/mixed blocks in some versions
    if hasattr(model, "stem") and hasattr(model, "repeat") and hasattr(model, "mixed_6a"):
        stem = model.stem
        A = []
        if hasattr(model, "mixed_5b"):
            A.append(getattr(model, "mixed_5b"))
        A.extend(list(getattr(model, "repeat", nn.Sequential()).children()))
        redA = getattr(model, "mixed_6a")
        B = list(getattr(model, "repeat_1", nn.Sequential()).children())
        redB = getattr(model, "mixed_7a")
        C = list(getattr(model, "repeat_2", nn.Sequential()).children())
        # include final high-level conv if exposed
        if hasattr(model, "block8"):
            C.append(getattr(model, "block8"))
        if hasattr(model, "conv2d_7b"):
            C.append(getattr(model, "conv2d_7b"))
        return stem, A, redA, B, redB, C

    # fallback: stem + flat blocks list
    stem = getattr(model, "stem", None)
    blocks = list(getattr(model, "blocks", []))
    if stem is None and all(hasattr(model, n) for n in ("conv2d_1a", "conv2d_2a", "conv2d_2b", "maxpool_3a", "conv2d_3b", "conv2d_4a", "maxpool_5a")):
        stem = nn.Sequential(
            model.conv2d_1a,
            model.conv2d_2a,
            model.conv2d_2b,
            model.maxpool_3a,
            model.conv2d_3b,
            model.conv2d_4a,
            model.maxpool_5a,
        )

    if not blocks:
        blocks = []
        if hasattr(model, "mixed_5b"):
            blocks.append(getattr(model, "mixed_5b"))
        blocks.extend(list(getattr(model, "repeat", [])))
        redA = getattr(model, "mixed_6a", getattr(model, "reduction_a", None))
        blocks_B = list(getattr(model, "repeat_1", []))
        redB = getattr(model, "mixed_7a", getattr(model, "reduction_b", None))
        blocks_C = list(getattr(model, "repeat_2", []))
        if hasattr(model, "block8"):
            blocks_C.append(getattr(model, "block8"))
        if hasattr(model, "conv2d_7b"):
            blocks_C.append(getattr(model, "conv2d_7b"))
        if blocks and blocks_B and blocks_C and redA is not None and redB is not None:
            return stem, list(blocks), redA, list(blocks_B), redB, list(blocks_C)

    if stem is None or not blocks:
        raise AttributeError("Unexpected InceptionResNetV2 structure; missing 'features'/'stem'/'blocks'.")

    A: list[nn.Module] = []
    B: list[nn.Module] = []
    C: list[nn.Module] = []
    redA = redB = None
    current = A

    for blk in blocks:
        name = blk.__class__.__name__.lower()
        if "mixed_6a" in name or "reductiona" in name or "reduction_a" in name:
            redA = blk
            current = B
            continue
        if "mixed_7a" in name or "reductionb" in name or "reduction_b" in name:
            redB = blk
            current = C
            continue
        current.append(blk)

    if redA is None or redB is None:
        raise ValueError("Could not locate reduction blocks in InceptionResNetV2 fallback split.")
    return stem, A, redA, B, redB, C


class TianMSDNet(nn.Module):
    """
    Tian Multi-scale Dense Inception backbone + head.

    Variants:
      - msd_inception_v4
      - msd_inception_resnet_v2
    """

    def __init__(
        self,
        *,
        variant: str,
        num_classes: int,
        growth: int = 128,
        dropout: float = 0.8,
        pretrained: bool = True,
        proj_stem: bool = True,
        proj_stages: bool = True,
    ):
        super().__init__()

        if variant == "msd_inception_v4":
            backbone = timm.create_model("inception_v4", pretrained=pretrained, num_classes=0)
            stem, A, redA, B, redB, C = _split_inception_v4(backbone)
        elif variant == "msd_inception_resnet_v2":
            backbone = timm.create_model("inception_resnet_v2", pretrained=pretrained, num_classes=0)
            stem, A, redA, B, redB, C = _split_inception_resnet_v2(backbone)
        else:
            raise ValueError("variant must be 'msd_inception_v4' or 'msd_inception_resnet_v2'")

        self.variant = variant
        self.stem = stem
        self.redA = redA
        self.redB = redB

        self.stageA = DenseRepeatStage(nn.ModuleList(A), growth)
        self.stageB = DenseRepeatStage(nn.ModuleList(B), growth)
        self.stageC = DenseRepeatStage(nn.ModuleList(C), growth)

        self.head: MultiScaleHead | None = None
        self.num_classes = num_classes
        self.dropout = dropout
        self.proj_stem = proj_stem
        self.proj_stages = proj_stages
        self.feature_dim: int | None = None

    def _forward_backbone(self, x: torch.Tensor):
        stem = self.stem(x)
        outA = self.stageA(stem)
        x = self.redA(self.stageA.last_cur if self.stageA.last_cur is not None else outA)
        outB = self.stageB(x)
        x = self.redB(self.stageB.last_cur if self.stageB.last_cur is not None else outB)
        outC = self.stageC(x)
        return stem, outA, outB, outC

    def _get_head(self, stem: torch.Tensor, outA: torch.Tensor, outB: torch.Tensor, outC: torch.Tensor) -> MultiScaleHead:
        if self.head is None:
            self.head = MultiScaleHead(
                stem_ch=stem.shape[1],
                stage_chs=[outA.shape[1], outB.shape[1], outC.shape[1]],
                num_classes=self.num_classes,
                dropout=self.dropout,
                proj_stem=self.proj_stem,
                proj_stages=self.proj_stages,
            ).to(stem.device)
        return self.head

    def forward(self, x: torch.Tensor, return_feats: bool = False):
        stem, outA, outB, outC = self._forward_backbone(x)
        head = self._get_head(stem, outA, outB, outC)
        if return_feats:
            logits, feats = head(stem, [outA, outB, outC], return_feats=True)
            self.feature_dim = feats.shape[1]
            return logits, feats
        return head(stem, [outA, outB, outC])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _, feats = self.forward(x, return_feats=True)
        return feats
