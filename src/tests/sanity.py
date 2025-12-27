"""Lightweight sanity checks for dataloaders and fusion variants."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.dataloader import HIPPADataLoader  # noqa: E402
from models.model_builder import build_model  # noqa: E402
from training.training import _forward_factory  # noqa: E402


def _make_loader(data_root: Path, csv_path: Path, batch_size: int = 2) -> HIPPADataLoader:
    return HIPPADataLoader(
        image_dir=data_root,
        csv_path=csv_path,
        batch_size=batch_size,
        num_workers=0,
        drop_last=False,
        strict_csv_match=False,
        use_trivial_augment=False,
    )


def check_dataloaders(data_root: Path, csv_path: Path) -> Dict[str, tuple]:
    """Return shapes from the first batch of each loader variant."""
    data = _make_loader(data_root, csv_path)
    loader_map = {
        "img": data.dataloader_image,
        "img_cult": data.dataloader_image_cultivar,
        "img_prog": data.dataloader_image_progression,
        "img_prog_cult": data.dataloader_image_progression_cultivar,
    }
    shapes = {}
    for name, fn in loader_map.items():
        batch = next(iter(fn("train")))
        parts = tuple(t.shape for t in batch)
        shapes[name] = parts
        print(f"[dataloader] {name}: {parts}")
    return shapes


def check_forward_passes(data_root: Path, csv_path: Path, device: torch.device | None = None) -> None:
    """Run one forward pass for each variant/fusion mode with a single batch."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _make_loader(data_root, csv_path, batch_size=2)
    num_classes = len(data.label2id)
    fusion_modes = ("concat", "sum", "prod", "concat+sum+prod")
    variants = [{"name": "conv4_pure", "model": "conv4dcnn", "mode": "img", "fusion_mode": None}]
    for fm in fusion_modes:
        variants.extend([
            {"name": f"conv4_cult_{fm}", "model": "conv4dcnn_latefusion", "mode": "img_cult", "fusion_mode": fm},
            {"name": f"conv4_prog_{fm}", "model": "conv4dcnn_latefusion", "mode": "img_prog", "fusion_mode": fm},
            {"name": f"conv4_both_{fm}", "model": "conv4dcnn_latefusion", "mode": "img_prog_cult", "fusion_mode": fm},
        ])
    loader_map = {
        "img": data.dataloader_image,
        "img_cult": data.dataloader_image_cultivar,
        "img_prog": data.dataloader_image_progression,
        "img_prog_cult": data.dataloader_image_progression_cultivar,
    }
    forward_map = {
        "img": _forward_factory("img"),
        "img_cult": _forward_factory("img_cult"),
        "img_prog": _forward_factory("img_prog"),
        "img_prog_cult": _forward_factory("img_prog_cult"),
    }

    for var in variants:
        batch = next(iter(loader_map[var["mode"]]("train")))
        kwargs = dict(
            model_name=var["model"],
            in_channels=3,
            num_classes=num_classes,
            img_size=data.img_size,
            hidden_dim=128,
            dropout=0.5,
        )
        if var["fusion_mode"] is not None:
            kwargs["fusion_mode"] = var["fusion_mode"]
        if "cult" in var["mode"]:
            kwargs["num_cultivars"] = len(data.cultivar2id)
        if "prog" in var["mode"]:
            kwargs["num_progressions"] = len(data.progression2id)

        model = build_model(**kwargs).to(device)
        model.eval()
        with torch.no_grad():
            logits, labels = forward_map[var["mode"]](model, batch, device)
        print(f"[forward] {var['name']}: logits {tuple(logits.shape)}, labels {tuple(labels.shape)}")


if __name__ == "__main__":
    root = Path("/home/hemamgholizadeh/hippa-rgb/data")
    csv = root / "meta_data.csv"
    check_dataloaders(root, csv)
    check_forward_passes(root, csv)
