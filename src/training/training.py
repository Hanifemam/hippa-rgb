"""Train an image classifier on the HIPPA RGB dataset."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from data.dataloader import HIPPADataLoader  # noqa: E402
from models.model_builder import build_model  # noqa: E402

try:  # Prefer package-relative import; fallback to local module when run as a script.
    from training.engine import train  # type: ignore
except Exception:  # pragma: no cover
    from engine import train  # type: ignore


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    data_root = Path("/home/hemamgholizadeh/hippa-rgb/data")
    csv_path = data_root / "meta_data.csv"
    results_root = Path("/mnt/slowdisk/public/HIPPA/multiclass_classification/results")
    run_dir = results_root / datetime.now().strftime("%Y%m%d-%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparams = {
        "model_name": "conv4dcnn",
        "image_dir": str(data_root),
        "csv_path": str(csv_path),
        "batch_size": 32,
        "img_size": 224,
        "in_channels": 3,
        "hidden_dim": 512,
        "dropout": 0.5,
        "num_workers": 0,  # avoid worker hangs during debug
        "pin_memory": False,
        "drop_last": False,
        "strict_csv_match": False,
        "learning_rate": 1e-4,
        "epochs": 10,
        "optimizer": "Adam",
        "loss_fn": "CrossEntropyLoss",
        "device": str(device),
    }

    data = HIPPADataLoader(
        image_dir=data_root,
        csv_path=csv_path,
        batch_size=hyperparams["batch_size"],
        img_size=hyperparams["img_size"],
        num_workers=hyperparams["num_workers"],
        pin_memory=hyperparams["pin_memory"],
        drop_last=hyperparams["drop_last"],
        strict_csv_match=hyperparams["strict_csv_match"],
    )
    train_loader = data.dataloader_image("train")
    val_loader = data.dataloader_image("val")

    # Quick sanity check to surface dataloader issues early.
    try:
        first_batch = next(iter(train_loader))
        x0, y0 = first_batch
        print(f"[Sanity] First batch loaded: images {tuple(x0.shape)}, labels {tuple(y0.shape)}")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Dataloader failed to produce a batch: {exc}") from exc

    model = build_model(
        model_name=hyperparams["model_name"],
        in_channels=hyperparams["in_channels"],
        num_classes=len(data.label2id),
        img_size=hyperparams["img_size"],
        hidden_dim=hyperparams["hidden_dim"],
        dropout=hyperparams["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss()

    history = train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=hyperparams["epochs"],
        device=device,
    )

    artifacts = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": data.label2id,
        "hyperparameters": hyperparams,
        "history": history,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts, run_dir / "hippa_rgb_classifier.pth")
    _write_json(run_dir / "history.json", history)

    summary_lines = [
        "HIPPA RGB classifier training run",
        f"Run directory: {run_dir}",
        "",
        "Hyperparameters:",
    ]
    summary_lines.extend([f"- {k}: {v}" for k, v in hyperparams.items()])
    if history.get("val_acc"):
        summary_lines.append(f"\nFinal val_acc: {history['val_acc'][-1]:.4f}")
    _write_text(run_dir / "summary.txt", "\n".join(summary_lines))

    print("Training complete. Artifacts saved to", run_dir)


if __name__ == "__main__":
    main()
