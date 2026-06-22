from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from data.dataloader import HIPPADataLoader
from training.run_final_experiments import _prepare_true_metadata


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(path)


def _tensor_transform(_: Image.Image) -> torch.Tensor:
    return torch.ones(3, 8, 8)


def test_hippa_dataloader_exposes_all_batch_shapes(tmp_path: Path) -> None:
    data_root = tmp_path / "disease"
    names = []
    for split in ("train", "val"):
        for class_name in ("healthy", "rust"):
            name = f"{split}_{class_name}.png"
            _write_image(data_root / split / class_name / name)
            names.append(name)

    metadata = pd.DataFrame(
        {
            "sample_name": names,
            "cultivar": ["gala", "fuji", "gala", "fuji"],
            "progression": ["early", "late", "early", "late"],
        }
    )
    csv_path = tmp_path / "metadata.csv"
    metadata.to_csv(csv_path, index=False)
    data = HIPPADataLoader(
        image_dir=data_root,
        csv_path=csv_path,
        batch_size=2,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        train_tf=_tensor_transform,
        eval_tf=_tensor_transform,
    )

    assert len(next(iter(data.dataloader_image("train")))) == 2
    assert len(next(iter(data.dataloader_image_cultivar("train")))) == 3
    assert len(next(iter(data.dataloader_image_progression("train")))) == 3
    assert len(next(iter(data.dataloader_image_progression_cultivar("train")))) == 4
    assert data.label2id == {"healthy": 0, "rust": 1}
    assert data.cultivar2id == {"__UNK__": 0, "fuji": 1, "gala": 2}


def test_dataloader_rejects_missing_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "disease"
    _write_image(data_root / "train" / "healthy" / "missing.png")
    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        columns=["sample_name", "cultivar", "progression"]
    ).to_csv(csv_path, index=False)

    with pytest.raises(KeyError, match="CSV missing 1 train image"):
        HIPPADataLoader(image_dir=data_root, csv_path=csv_path)


def test_prepare_true_metadata_filters_and_sorts_disease_images(tmp_path: Path) -> None:
    disease_root = tmp_path / "disease"
    _write_image(disease_root / "train" / "healthy" / "b.png")
    _write_image(disease_root / "val" / "rust" / "a.png")
    source = tmp_path / "source.csv"
    pd.DataFrame(
        [
            {"sample_name": "unused.png", "cultivar": "x", "progression": "none"},
            {"sample_name": "b.png", "cultivar": "gala", "progression": "early"},
            {"sample_name": "a.png", "cultivar": "fuji", "progression": "late"},
        ]
    ).to_csv(source, index=False)

    output = _prepare_true_metadata(source, disease_root, tmp_path / "out" / "true.csv")

    actual = pd.read_csv(output)
    assert actual["sample_name"].tolist() == ["a.png", "b.png"]
    assert actual.columns.tolist() == ["sample_name", "cultivar", "progression"]
