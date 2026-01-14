"""Progression-stage prediction with a ResNet backbone + cultivar late fusion."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms as T
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ploter import LateFusionClassifier, ResultSaver, fit_with_saving  # type: ignore

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().with_name("results_fusion")
DEFAULT_DATASET = REPO_ROOT / "data" / "progression"
DEFAULT_CULTIVAR_CSV = REPO_ROOT / "data" / "meta_data_cultivar.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RESNET_BUILDERS = {
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
    "resnet101": tv_models.resnet101,
    "resnet152": tv_models.resnet152,
}
RESNET_WEIGHTS = {
    "resnet18": tv_models.ResNet18_Weights.DEFAULT,
    "resnet34": tv_models.ResNet34_Weights.DEFAULT,
    "resnet50": tv_models.ResNet50_Weights.DEFAULT,
    "resnet101": tv_models.ResNet101_Weights.DEFAULT,
    "resnet152": tv_models.ResNet152_Weights.DEFAULT,
}
_TENSORBOARD_WARNED = False


def _resnet_sort_key(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


def _format_float(value: float) -> str:
    return f"{value:.0e}"


def _tf(train: bool, img_size: int) -> T.Compose:
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize(int(img_size * 256 / 224)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _image_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def _load_cultivar_map(csv_path: Path, image_col: str = "sample_name", cultivar_col: str = "cultivar"):
    df = pd.read_csv(csv_path)
    for col in (image_col, cultivar_col):
        if col not in df.columns:
            raise KeyError(f"CSV missing '{col}'. Found: {list(df.columns)}")
    df[image_col] = df[image_col].astype(str)
    df[cultivar_col] = df[cultivar_col].astype(str)
    cultivars = sorted(df[cultivar_col].dropna().unique().tolist())
    cultivar2id = {name: idx for idx, name in enumerate(cultivars)}
    sample_to_cultivar = {row[image_col]: cultivar2id[row[cultivar_col]] for row in df.to_dict("records")}
    return sample_to_cultivar, cultivar2id


def _class_names(train_dir: Path) -> tuple[list[str], dict[str, int]]:
    class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found under {train_dir}")
    class_names = []
    class_to_idx: dict[str, int] = {}
    for idx, d in enumerate(class_dirs):
        name = d.name
        class_to_idx[name] = idx
        class_names.append(name.replace("class_", "", 1) if name.startswith("class_") else name)
    return class_names, class_to_idx


class ProgressionCultivarDataset(Dataset):
    def __init__(
        self,
        *,
        root: Path,
        split: str,
        class_to_idx: dict[str, int],
        cultivar_map: dict[str, int],
        tfm: T.Compose,
        strict: bool = True,
    ):
        self.tfm = tfm
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.items: list[tuple[Path, int, int, str]] = []
        missing = []
        for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            if class_dir.name not in class_to_idx:
                raise ValueError(f"Unexpected class folder '{class_dir.name}' under {split_dir}")
            label = class_to_idx[class_dir.name]
            for p in _image_files(class_dir):
                name = p.name
                if name not in cultivar_map:
                    missing.append(name)
                    continue
                self.items.append((p, label, cultivar_map[name], name))

        if strict and missing:
            missing = sorted(set(missing))
            raise KeyError(f"Missing cultivar labels for {len(missing)} file(s), e.g. {missing[:5]}")
        if not self.items:
            raise ValueError(f"No images found under {split_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, cultivar_id, name = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.tfm(img)
        return img, torch.tensor(label, dtype=torch.long), torch.tensor(cultivar_id, dtype=torch.long), name


def make_progression_loaders(
    data_directory: str | Path,
    cultivar_csv: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    img_size: int,
    strict: bool = True,
):
    data_root = Path(data_directory)
    cultivar_csv = Path(cultivar_csv)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    if not cultivar_csv.exists():
        raise FileNotFoundError(f"Cultivar CSV not found: {cultivar_csv}")

    class_names, class_to_idx = _class_names(data_root / "train")
    cultivar_map, cultivar2id = _load_cultivar_map(cultivar_csv)
    train_tf = _tf(True, img_size)
    eval_tf = _tf(False, img_size)

    train_ds = ProgressionCultivarDataset(
        root=data_root,
        split="train",
        class_to_idx=class_to_idx,
        cultivar_map=cultivar_map,
        tfm=train_tf,
        strict=strict,
    )
    val_ds = ProgressionCultivarDataset(
        root=data_root,
        split="val",
        class_to_idx=class_to_idx,
        cultivar_map=cultivar_map,
        tfm=eval_tf,
        strict=strict,
    )

    test_loader = None
    test_dir = data_root / "test"
    if test_dir.exists():
        test_ds = ProgressionCultivarDataset(
            root=data_root,
            split="test",
            class_to_idx=class_to_idx,
            cultivar_map=cultivar_map,
            tfm=eval_tf,
            strict=strict,
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if test_loader is None:
        print("[late_fusion] test split not found; using val loader for test metrics.")
        test_loader = val_loader

    return train_loader, val_loader, test_loader, class_names, len(cultivar2id)


def _configure_backbone_trainable(backbone: nn.Module, unfreeze_layers: int) -> None:
    if unfreeze_layers < 0:
        for param in backbone.parameters():
            param.requires_grad = True
        return
    for param in backbone.parameters():
        param.requires_grad = False
    stages = []
    for name in ("layer4", "layer3", "layer2", "layer1", "conv1", "bn1"):
        if hasattr(backbone, name):
            stages.append(getattr(backbone, name))
    for stage in stages[: max(0, unfreeze_layers)]:
        for param in stage.parameters():
            param.requires_grad = True


def get_backbone(resnet_name: str, *, unfreeze_layers: int, pretrained: bool) -> tuple[nn.Module, int]:
    name = resnet_name.lower()
    if name not in RESNET_BUILDERS:
        raise ValueError(f"Unknown ResNet '{resnet_name}'. Available: {sorted(RESNET_BUILDERS)}")
    weights = RESNET_WEIGHTS[name] if pretrained else None
    backbone = RESNET_BUILDERS[name](weights=weights)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    _configure_backbone_trainable(backbone, unfreeze_layers)
    return backbone, feat_dim


def train_progression_late_fusion(
    data_directory: str | Path = DEFAULT_DATASET,
    *,
    cultivar_csv: str | Path = DEFAULT_CULTIVAR_CSV,
    epochs: int = 20,
    early_stopping_patience: Optional[int] = 5,
    early_stopping_min_delta: float = 0.0,
    unfreeze_layers: int = -1,
    fusion_mode: str = "concat+sum+prod",
    batch_size: int = 32,
    num_workers: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    img_size: int = 224,
    resnet_name: str = "resnet18",
    pretrained: bool = True,
    device: Optional[str] = None,
    save_base_dir: str | Path = DEFAULT_RESULTS_ROOT,
):
    """
    Train a late-fusion model that combines RGB features with cultivar embeddings.
    """
    global _TENSORBOARD_WARNED
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader, test_loader, class_names, num_types = make_progression_loaders(
        data_directory,
        cultivar_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
    )
    if train_loader is None or val_loader is None or test_loader is None:
        raise RuntimeError("Missing one of the required splits: train/val/test.")

    backbone, feat_dim = get_backbone(resnet_name, unfreeze_layers=unfreeze_layers, pretrained=pretrained)
    backbone = backbone.to(dev)

    head = LateFusionClassifier(
        feat_dim=feat_dim,
        num_classes=len(class_names),
        num_types=num_types,
        fusion_mode=fusion_mode,
    ).to(dev)

    params = [p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    run_name = (
        f"prog-rgb-apple-{resnet_name}-{fusion_mode.replace('+', '_')}"
        f"-lr{_format_float(lr)}-wd{_format_float(weight_decay)}"
        f"-{datetime.now():%Y%m%d-%H%M%S}"
    )
    saver = ResultSaver(base_dir=str(save_base_dir), run_name=run_name)
    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(saver.base_dir / "tensorboard"))
    else:
        if not _TENSORBOARD_WARNED:
            print("[late_fusion] TensorBoard not available; skipping SummaryWriter logging.")
            _TENSORBOARD_WARNED = True

    results = fit_with_saving(
        backbone=backbone,
        head=head,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=dev,
        epochs=epochs,
        optimizer=optimizer,
        class_names=class_names,
        input_shape=(feat_dim,),
        image_shape=(3, img_size, img_size),
        saver=saver,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        writer=writer,
    )
    if writer is not None:
        writer.close()

    results.update(
        {
            "class_names": class_names,
            "num_apple_types": num_types,
            "num_cultivars": num_types,
            "fusion_mode": fusion_mode,
            "resnet": resnet_name,
            "pretrained": pretrained,
            "lr": lr,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "save_dir": str(saver.base_dir),
            "run_name": run_name,
        }
    )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train progression-stage late fusion model.")
    parser.add_argument("--data", default=str(DEFAULT_DATASET), help="Root of prepared dataset.")
    parser.add_argument("--cultivar-csv", default=str(DEFAULT_CULTIVAR_CSV), help="CSV with cultivar labels.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--img-size", type=int, default=224, help="Image resize.")
    parser.add_argument(
        "--resnet",
        nargs="+",
        default=list(RESNET_BUILDERS),
        choices=sorted(RESNET_BUILDERS),
        help="ResNet backbone variant(s) for grid search.",
    )
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained weights.")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Disable pretrained weights.")
    parser.add_argument(
        "--fusion-mode",
        default="all",
        help="Fusion mode (sum, prod, concat, concat+sum+prod) or 'all' to run every mode.",
    )
    parser.add_argument("--unfreeze-layers", type=int, default=-1, help="Backbone unfreeze depth.")
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=[3e-4, 1e-4, 3e-5],
        help="Learning rate grid.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        nargs="+",
        default=[1e-4],
        help="Weight decay grid.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Early stopping patience on val loss (<=0 disables).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum val-loss improvement to reset early stopping.",
    )
    parser.add_argument("--device", default=None, help="Override device (e.g., 'cpu').")
    parser.add_argument(
        "--save-dir",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Directory for training outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    requested = args.fusion_mode.lower()
    modes = (
        ["sum", "prod", "concat", "concat+sum+prod"]
        if requested == "all"
        else [args.fusion_mode]
    )

    resnet_grid = sorted(args.resnet, key=_resnet_sort_key)
    lr_grid = args.lr
    weight_decay_grid = args.weight_decay
    total = len(resnet_grid) * len(lr_grid) * len(weight_decay_grid) * len(modes)
    run_idx = 0

    print(
        "[late_fusion] Grid search config:"
        f" resnets={resnet_grid}, lrs={lr_grid}, weight_decays={weight_decay_grid}, modes={modes}"
    )

    for resnet_name in resnet_grid:
        for lr in lr_grid:
            for weight_decay in weight_decay_grid:
                for mode in modes:
                    run_idx += 1
                    print(
                        "[late_fusion] "
                        f"({run_idx}/{total}) resnet={resnet_name} lr={lr} "
                        f"weight_decay={weight_decay} fusion_mode='{mode}'"
                    )
                    train_progression_late_fusion(
                        data_directory=args.data,
                        cultivar_csv=args.cultivar_csv,
                        epochs=args.epochs,
                        early_stopping_patience=args.early_stop_patience,
                        early_stopping_min_delta=args.early_stop_min_delta,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        img_size=args.img_size,
                        fusion_mode=mode,
                        unfreeze_layers=args.unfreeze_layers,
                        resnet_name=resnet_name,
                        pretrained=args.pretrained,
                        lr=lr,
                        weight_decay=weight_decay,
                        device=args.device,
                        save_base_dir=args.save_dir,
                    )


if __name__ == "__main__":
    main()
