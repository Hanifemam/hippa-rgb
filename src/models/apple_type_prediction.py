"""Train the apple-type classifier and export cultivar metadata predictions."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "data" / "apple_type"
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().with_name("results_apple_type")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _transform(train: bool, img_size: int) -> T.Compose:
    steps = [T.Resize((img_size, img_size))]
    if train:
        steps.extend([T.RandomHorizontalFlip(), T.RandomRotation(10)])
    steps.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose(steps)


def _image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _class_names(train_dir: Path) -> list[str]:
    names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if not names:
        raise ValueError(f"No class folders found under {train_dir}")
    return names


class AppleTypeDataset(Dataset):
    def __init__(self, root: Path, split: str, class_to_id: dict[str, int], transform: T.Compose):
        self.transform = transform
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        self.items: list[tuple[Path, int, str]] = []
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if class_dir.name not in class_to_id:
                raise ValueError(f"Unexpected class folder '{class_dir.name}' under {split_dir}")
            self.items.extend(
                (path, class_to_id[class_dir.name], path.name) for path in _image_files(class_dir)
            )
        if not self.items:
            raise ValueError(f"No images found under {split_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label, name = self.items[index]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, torch.tensor(label, dtype=torch.long), name


class PredictionDataset(Dataset):
    def __init__(self, root: Path, transform: T.Compose):
        self.transform = transform
        self.items = _image_files(root)
        if not self.items:
            raise ValueError(f"No images found under {root}")
        names = [path.name for path in self.items]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            raise ValueError(f"Duplicate sample names found, e.g. {sorted(duplicates)[:5]}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path = self.items[index]
        return self.transform(Image.open(path).convert("RGB")), path.name


class ResNet152Classifier(nn.Module):
    """Architecture used by the supplied apple-type checkpoint."""

    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 512,
        unfreeze_layers: int = 0,
    ):
        super().__init__()
        weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet152(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._set_trainable_layers(unfreeze_layers)

    def _set_trainable_layers(self, unfreeze_layers: int) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if unfreeze_layers == -1:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
            return
        stages = list(self.backbone.children())[::-1]
        for stage in stages[: max(0, unfreeze_layers)]:
            for parameter in stage.parameters():
                parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(images))


def _make_loaders(
    root: Path, batch_size: int, num_workers: int, img_size: int
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    class_names = _class_names(root / "train")
    class_to_id = {name: index for index, name in enumerate(class_names)}
    loaders = []
    for split in ("train", "val", "test"):
        dataset = AppleTypeDataset(
            root, split, class_to_id, _transform(split == "train", img_size)
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        )
    return loaders[0], loaders[1], loaders[2], class_names


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def save_cultivar_predictions(
    model: nn.Module,
    image_root: str | Path,
    class_names: list[str],
    output_csv: str | Path,
    *,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 1,
    img_size: int = 224,
) -> Path:
    dataset = PredictionDataset(Path(image_root), _transform(False, img_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    rows: list[tuple[str, str]] = []
    model.eval()
    with torch.no_grad():
        for images, names in loader:
            predictions = model(images.to(device)).argmax(1).cpu().tolist()
            rows.extend((name, class_names[prediction]) for name, prediction in zip(names, predictions))

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_name", "cultivar"])
        writer.writerows(rows)
    print(f"[save] {output}")
    return output


def train_apple_type(
    data_directory: str | Path = DEFAULT_DATASET,
    *,
    prediction_data: str | Path | None = None,
    output_csv: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    num_workers: int = 1,
    img_size: int = 224,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    unfreeze_layers: int = 0,
    patience: int = 5,
    pretrained: bool = True,
    device: Optional[str] = None,
    save_base_dir: str | Path = DEFAULT_RESULTS_ROOT,
) -> dict[str, object]:
    random.seed(42)
    torch.manual_seed(42)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = Path(data_directory)
    train_loader, val_loader, test_loader, class_names = _make_loaders(
        data_root, batch_size, num_workers, img_size
    )
    model = ResNet152Classifier(
        len(class_names), pretrained=pretrained, unfreeze_layers=unfreeze_layers
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    run_dir = Path(save_base_dir) / f"resnet152-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy, stale_epochs = -1.0, 0
    checkpoint = run_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, dev, optimizer)
        val_loss, val_accuracy = _run_epoch(model, val_loader, dev)
        scheduler.step(val_loss)
        print(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy, stale_epochs = val_accuracy, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                break

    model.load_state_dict(torch.load(checkpoint, map_location=dev))
    test_loss, test_accuracy = _run_epoch(model, test_loader, dev)
    csv_path = Path(output_csv) if output_csv else run_dir / "meta_data_cultivar.csv"
    save_cultivar_predictions(
        model,
        prediction_data or data_root,
        class_names,
        csv_path,
        device=dev,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
    )
    return {
        "model": model,
        "class_names": class_names,
        "best_val_accuracy": best_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "checkpoint": str(checkpoint),
        "prediction_csv": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train apple-type prediction and export cultivar CSV.")
    parser.add_argument("--data", default=str(DEFAULT_DATASET))
    parser.add_argument("--predict-data", default=None, help="Images that need cultivar predictions.")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--save-dir", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze-layers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    args = parser.parse_args()
    train_apple_type(
        args.data,
        prediction_data=args.predict_data,
        output_csv=args.output_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        unfreeze_layers=args.unfreeze_layers,
        patience=args.patience,
        pretrained=args.pretrained,
        device=args.device,
        save_base_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
