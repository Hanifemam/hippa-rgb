"""Metadata embedding + late-fusion utilities (cultivar/progression aware)."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn

FusionMode = Literal["concat", "sum", "prod", "concat+sum+prod"]


class CultivarEmbedding(nn.Module):
    """Embedding + LayerNorm for cultivar IDs."""
    def __init__(self, num_cultivars: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_cultivars, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embed(ids))


class ProgressionEmbedding(nn.Module):
    """Embedding + LayerNorm for progression IDs."""
    def __init__(self, num_progressions: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_progressions, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embed(ids))


class LateFusionCombiner(nn.Module):
    """
    Generic late fusion combiner: takes image features plus optional cultivar/progression embeddings.
    Returns fused features; attach any classifier head you like.
    """
    def __init__(self, feat_dim: int, fusion_mode: FusionMode = "concat+sum+prod"):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.img_norm = nn.LayerNorm(feat_dim)

    def _fuse(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.fusion_mode == "concat":
            return torch.cat(parts, dim=1)
        if self.fusion_mode == "sum":
            return torch.stack(parts, dim=0).sum(dim=0)
        if self.fusion_mode == "prod":
            return torch.stack(parts, dim=0).prod(dim=0)
        if self.fusion_mode == "concat+sum+prod":
            cat = torch.cat(parts, dim=1)
            summed = torch.stack(parts, dim=0).sum(dim=0)
            prod = torch.stack(parts, dim=0).prod(dim=0)
            return torch.cat([cat, summed, prod], dim=1)
        raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

    def forward(
        self,
        img_feats: torch.Tensor,
        *,
        cultivar_feats: Optional[torch.Tensor] = None,
        progression_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [self.img_norm(img_feats)]
        if cultivar_feats is not None:
            parts.append(cultivar_feats)
        if progression_feats is not None:
            parts.append(progression_feats)
        return self._fuse(parts)


class LateFusionHead(nn.Module):
    """
    Ready-to-use head: embeds cultivar/progression, fuses with image feats, classifies.
    Works with any backbone that outputs vectors of size feat_dim.
    """
    def __init__(
        self,
        *,
        feat_dim: int,
        num_classes: int,
        num_cultivars: Optional[int] = None,
        num_progressions: Optional[int] = None,
        fusion_mode: FusionMode = "concat+sum+prod",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_cultivar = num_cultivars is not None
        self.use_progression = num_progressions is not None
        self.cultivar_emb = CultivarEmbedding(num_cultivars, feat_dim) if self.use_cultivar else None
        self.progression_emb = ProgressionEmbedding(num_progressions, feat_dim) if self.use_progression else None
        self.combiner = LateFusionCombiner(feat_dim=feat_dim, fusion_mode=fusion_mode)

        in_dim = {
            "concat": feat_dim * (1 + int(self.use_cultivar) + int(self.use_progression)),
            "sum": feat_dim,
            "prod": feat_dim,
            "concat+sum+prod": feat_dim * (1 + int(self.use_cultivar) + int(self.use_progression)) + feat_dim * 2,
        }[fusion_mode]
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(
        self,
        img_feats: torch.Tensor,
        *,
        cultivar_ids: Optional[torch.Tensor] = None,
        progression_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cultivar_feats = self.cultivar_emb(cultivar_ids) if self.use_cultivar else None
        progression_feats = self.progression_emb(progression_ids) if self.use_progression else None
        fused = self.combiner(
            img_feats,
            cultivar_feats=cultivar_feats,
            progression_feats=progression_feats,
        )
        return self.classifier(fused)
