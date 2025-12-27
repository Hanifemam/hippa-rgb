"""Train an image classifier on the HIPPA RGB dataset."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from sklearn.metrics import confusion_matrix

try:  # Headless-safe plotting
    import matplotlib
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore

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
    print(f"[save] {path}")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[save] {path}")


def _build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    if name in {"sgd", "sgd+momentum", "sgd_momentum"}:
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")


def run_single_experiment(
    combo_dir: Path,
    base_hparams: Dict[str, Any],
    device: torch.device,
    train_loader,
    val_loader,
    forward_fn=None,
) -> Dict[str, Any]:
    """Train one hyperparameter combo and save artifacts."""
    combo_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = {
        "model_name": base_hparams["model_name"],
        "in_channels": base_hparams["in_channels"],
        "num_classes": base_hparams["num_classes"],
        "img_size": base_hparams["img_size"],
        "hidden_dim": base_hparams["hidden_dim"],
        "dropout": base_hparams["dropout"],
    }
    for opt_key in ("num_cultivars", "num_progressions", "fusion_mode"):
        if opt_key in base_hparams and base_hparams[opt_key] is not None:
            model_kwargs[opt_key] = base_hparams[opt_key]
    model = build_model(**model_kwargs).to(device)

    _save_model_summaries(
        model=model,
        combo_dir=combo_dir,
        in_channels=base_hparams["in_channels"],
        img_size=base_hparams["img_size"],
    )

    optimizer = _build_optimizer(base_hparams["optimizer"], model.parameters(), base_hparams["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    print("Hyperparameters for this run:")
    for k, v in base_hparams.items():
        print(f"  {k}: {v}")

    def _epoch_save(epoch_idx: int, history: Dict[str, List[float]], state_dict: Dict[str, torch.Tensor]):
        latest_path = combo_dir / "latest_checkpoint.pth"
        torch.save(
            {
                "epoch": epoch_idx,
                "model_state_dict": state_dict,
                "class_to_idx": base_hparams["class_to_idx"],
                "hyperparameters": base_hparams,
                "history": history,
            },
            latest_path,
        )
        print(f"[save] {latest_path}")
        _save_history(history, combo_dir / "history.json", combo_dir / "epoch_metrics.csv")

    train_result = train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=base_hparams["epochs"],
        device=device,
        scheduler=scheduler,
        scheduler_on="loss",
        early_stopping_patience=5,
        epoch_save_fn=_epoch_save,
        forward_fn=forward_fn,
    )

    # Load best weights for evaluation/pred logging.
    best_state = train_result.get("best_state_dict")
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save best model (if captured) or final.
    state_to_save = train_result.get("best_state_dict") or model.state_dict()
    artifacts = {
        "model_state_dict": state_to_save,
        "class_to_idx": base_hparams["class_to_idx"],
        "hyperparameters": base_hparams,
        "train_result": train_result,
    }
    ckpt_path = combo_dir / "hippa_rgb_classifier.pth"
    torch.save(artifacts, ckpt_path)
    print(f"[save] {ckpt_path}")

    history = train_result.get("history", {})
    _save_history(history, combo_dir / "history.json", combo_dir / "epoch_metrics.csv")

    # Save per-sample predictions on val split.
    preds_path = combo_dir / "val_predictions.csv"
    val_preds, val_labels, val_names = _save_predictions(
        model, val_loader, device, base_hparams["class_to_idx"], preds_path, forward_fn=forward_fn
    )

    _save_confusion_matrix(val_labels, val_preds, combo_dir / "val_confusion_matrix.png",
                           base_hparams["class_to_idx"])
    _save_learning_curves(history, combo_dir / "learning_curves.png")

    hist = train_result.get("history", {})
    summary_lines = [
        "HIPPA RGB classifier training run",
        f"Run directory: {combo_dir}",
        "",
        "Hyperparameters:",
    ]
    summary_lines.extend([f"- {k}: {v}" for k, v in base_hparams.items()])
    if hist.get("val_acc"):
        summary_lines.append(f"\nFinal val_acc: {hist['val_acc'][-1]:.4f}")
    if train_result.get("best_val_loss") is not None:
        summary_lines.append(f"Best val_loss: {train_result['best_val_loss']:.4f} at epoch {train_result.get('best_epoch', 'n/a')}")
    _write_text(combo_dir / "summary.txt", "\n".join(summary_lines))

    return train_result


def _save_predictions(model, loader, device, label2id: Dict[str, int], out_path: Path, forward_fn=None) -> Tuple[List[int], List[int], List[str]]:
    """Save sample_name, true_label, pred_label for the given loader."""
    id2label = {v: k for k, v in label2id.items()}
    records: List[List[str]] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_names: List[str] = []
    items = getattr(loader.dataset, "items", [])
    idx = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if forward_fn is None:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
            else:
                logits, labels = forward_fn(model, batch, device)
            preds = logits.argmax(dim=1).cpu().tolist()
            labels_list = labels.cpu().tolist()
            bs = len(labels_list)
            for j in range(bs):
                name = items[idx + j][2] if idx + j < len(items) else ""
                true_lbl = id2label.get(labels_list[j], labels_list[j])
                pred_lbl = id2label.get(preds[j], preds[j])
                records.append([name, true_lbl, pred_lbl])
                all_preds.append(preds[j])
                all_labels.append(labels_list[j])
                all_names.append(name)
            idx += bs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_name", "true_label", "pred_label"])
        writer.writerows(records)
    print(f"[save] {out_path}")
    return all_preds, all_labels, all_names


def _save_model_summaries(model, combo_dir: Path, in_channels: int, img_size: int) -> None:
    """Save detailed and brief model summaries."""
    detailed = None
    brief = repr(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + non_trainable
    counts_txt = (
        f"\n\n[Params] trainable: {trainable:,}, "
        f"non_trainable: {non_trainable:,}, total: {total:,}"
    )
    try:
        from torchinfo import summary  # type: ignore

        detailed = summary(
            model,
            input_size=(1, in_channels, img_size, img_size),
            depth=4,
            verbose=1,
        )
        detailed = str(detailed) + counts_txt

        brief_summary = summary(
            model,
            input_size=(1, in_channels, img_size, img_size),
            depth=2,
            verbose=0,
        )
        brief = str(brief_summary) + counts_txt
    except Exception as exc:  # pragma: no cover
        if detailed is None:
            detailed = f"torchinfo unavailable: {exc}\n\n{repr(model)}{counts_txt}"
        brief = brief + counts_txt

    combo_dir.mkdir(parents=True, exist_ok=True)
    _write_text(combo_dir / "model_summary_detailed.txt", detailed or brief)
    _write_text(combo_dir / "model_summary_brief.txt", brief)
    print(f"[save] {combo_dir / 'model_summary_detailed.txt'}")
    print(f"[save] {combo_dir / 'model_summary_brief.txt'}")


def _save_confusion_matrix(labels: List[int], preds: List[int], out_path: Path, label2id: Dict[str, int]) -> None:
    """Save confusion matrix plot if matplotlib is available."""
    if plt is None:
        return
    id2label = {v: k for k, v in label2id.items()}
    cm = confusion_matrix(labels, preds, labels=sorted(id2label))
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = [id2label[i] for i in sorted(id2label)]
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel="True label", xlabel="Predicted label",
           title="Val Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def _save_learning_curves(history: Dict[str, List[float]], out_path: Path) -> None:
    """Plot and save learning curves for loss/accuracy."""
    if plt is None or not history:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ["train_loss", "val_loss", "train_acc", "val_acc"]:
        if key in history and history[key]:
            ax.plot(history[key], label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Learning Curves")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def _save_history(history: Dict[str, List[float]], json_path: Path, csv_path: Path) -> None:
    """Persist history to JSON and CSV (train/val loss/acc)."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(json_path, history or {})
    if not history:
        return
    max_len = max(len(history.get("train_loss", [])), len(history.get("val_loss", [])),
                  len(history.get("train_acc", [])), len(history.get("val_acc", [])))
    rows = []
    for i in range(max_len):
        rows.append({
            "epoch": i + 1,
            "train_loss": history.get("train_loss", [None] * max_len)[i] if len(history.get("train_loss", [])) > i else None,
            "val_loss": history.get("val_loss", [None] * max_len)[i] if len(history.get("val_loss", [])) > i else None,
            "train_acc": history.get("train_acc", [None] * max_len)[i] if len(history.get("train_acc", [])) > i else None,
            "val_acc": history.get("val_acc", [None] * max_len)[i] if len(history.get("val_acc", [])) > i else None,
        })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {csv_path}")


def _forward_factory(mode: str):
    """Return a forward_fn compatible with engine.train for the given dataloader mode."""
    def _img(model, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return model(x), y

    def _cult(model, batch, device):
        x, cult, y = batch
        x, cult, y = x.to(device), cult.to(device), y.to(device)
        return model(x, cultivar_ids=cult), y

    def _prog(model, batch, device):
        x, prog, y = batch
        x, prog, y = x.to(device), prog.to(device), y.to(device)
        return model(x, progression_ids=prog), y

    def _both(model, batch, device):
        x, prog, cult, y = batch
        x, prog, cult, y = x.to(device), prog.to(device), cult.to(device), y.to(device)
        return model(x, progression_ids=prog, cultivar_ids=cult), y

    return {
        "img": _img,
        "img_cult": _cult,
        "img_prog": _prog,
        "img_prog_cult": _both,
    }[mode]


def main():
    data_root = Path("/home/hemamgholizadeh/hippa-rgb/data")
    csv_path = data_root / "meta_data.csv"
    results_root = Path("/mnt/slowdisk/public/HIPPA/multiclass_classification/results")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    worker_count = os.cpu_count()
    base_hparams = {
        "model_name": "conv4dcnn",
        "image_dir": str(data_root),
        "csv_path": str(csv_path),
        "batch_size": 32,
        "img_size": 224,
        "in_channels": 3,
        "hidden_dim": 512,
        "dropout": 0.5,
        "num_workers": worker_count, 
        "pin_memory": False,
        "drop_last": False,
        "strict_csv_match": False,
        "use_trivial_augment": True,
        "learning_rate": 1e-4,
        "epochs": 20,
        "optimizer": "Adam",
        "loss_fn": "CrossEntropyLoss",
        "device": str(device),
    }
    run_root = results_root / f"conv4dcnn_fusion_grid_{ts}"

    data = HIPPADataLoader(
        image_dir=data_root,
        csv_path=csv_path,
        batch_size=base_hparams["batch_size"],
        img_size=base_hparams["img_size"],
        num_workers=base_hparams["num_workers"],
        pin_memory=base_hparams["pin_memory"],
        drop_last=base_hparams["drop_last"],
        strict_csv_match=base_hparams["strict_csv_match"],
        use_trivial_augment=base_hparams["use_trivial_augment"],
    )
    forward_map = {
        "img": _forward_factory("img"),
        "img_cult": _forward_factory("img_cult"),
        "img_prog": _forward_factory("img_prog"),
        "img_prog_cult": _forward_factory("img_prog_cult"),
    }
    loader_map = {
        "img": data.dataloader_image,
        "img_cult": data.dataloader_image_cultivar,
        "img_prog": data.dataloader_image_progression,
        "img_prog_cult": data.dataloader_image_progression_cultivar,
    }
    fusion_modes = ("concat", "sum", "prod", "concat+sum+prod")
    variants = [
        {"name": "conv4_pure", "model_name": "conv4dcnn", "mode": "img", "fusion_mode": None},
    ]
    for fm in fusion_modes:
        variants.extend([
            {"name": f"conv4_cult_{fm}", "model_name": "conv4dcnn_latefusion", "mode": "img_cult", "fusion_mode": fm},
            {"name": f"conv4_prog_{fm}", "model_name": "conv4dcnn_latefusion", "mode": "img_prog", "fusion_mode": fm},
            {"name": f"conv4_both_{fm}", "model_name": "conv4dcnn_latefusion", "mode": "img_prog_cult", "fusion_mode": fm},
        ])
    grid = {
        "dropout": [0.3],
        "learning_rate": [1e-2, 5e-4, 1e-4],
        "optimizer": ["Adam"],
    }
# "optimizer": ["Adam", "AdamW"],
    run_root.mkdir(parents=True, exist_ok=True)
    run_summaries: List[Dict[str, Any]] = []

    for variant in variants:
        mode = variant["mode"]
        fusion_mode = variant["fusion_mode"]
        train_loader = loader_map[mode]("train")
        val_loader = loader_map[mode]("val")
        for dropout in grid["dropout"]:
            for lr in grid["learning_rate"]:
                for opt_name in grid["optimizer"]:
                    combo_name = f"{variant['name']}_drop{dropout}_lr{lr}_opt{opt_name.replace('+','').lower()}"
                    combo_dir = run_root / combo_name
                    hparams = dict(base_hparams)
                    hparams.update({
                        "model_name": variant["model_name"],
                        "dropout": dropout,
                        "learning_rate": lr,
                        "optimizer": opt_name,
                        "class_to_idx": data.label2id,
                        "num_classes": len(data.label2id),
                    })
                    if fusion_mode is not None:
                        hparams["fusion_mode"] = fusion_mode
                    if "cult" in mode:
                        hparams["num_cultivars"] = len(data.cultivar2id)
                    if "prog" in mode:
                        hparams["num_progressions"] = len(data.progression2id)
                    print(f"\n=== Running combo: {combo_name} ===")
                    result = run_single_experiment(
                        combo_dir,
                        hparams,
                        device,
                        train_loader,
                        val_loader,
                        forward_map[mode],
                    )
                    run_summaries.append({
                        "combo": combo_name,
                        "variant": variant["name"],
                        "dir": str(combo_dir),
                        "best_val_loss": result.get("best_val_loss"),
                        "best_epoch": result.get("best_epoch"),
                        "final_val_acc": result.get("history", {}).get("val_acc", [])[-1] if result.get("history", {}).get("val_acc") else None,
                        "hparams": {"dropout": dropout, "learning_rate": lr, "optimizer": opt_name, "fusion_mode": fusion_mode},
                    })

    best_run = None
    loss_ranked = [r for r in run_summaries if r.get("best_val_loss") is not None]
    if loss_ranked:
        best_run = min(loss_ranked, key=lambda r: r["best_val_loss"])
    else:
        acc_ranked = [r for r in run_summaries if r.get("final_val_acc") is not None]
        if acc_ranked:
            best_run = max(acc_ranked, key=lambda r: r["final_val_acc"])

    _write_json(run_root / "grid_summary.json", run_summaries)

    if best_run:
        best_dir = Path(best_run["dir"])
        best_ckpt = best_dir / "hippa_rgb_classifier.pth"
        if best_ckpt.exists():
            dest_ckpt = run_root / "best_model.pth"
            shutil.copy2(best_ckpt, dest_ckpt)
            best_meta = {
                "combo": best_run["combo"],
                "source_checkpoint": str(best_ckpt),
                "best_val_loss": best_run.get("best_val_loss"),
                "best_epoch": best_run.get("best_epoch"),
                "final_val_acc": best_run.get("final_val_acc"),
            }
            _write_json(run_root / "best_model_meta.json", best_meta)
            print(f"[save] {dest_ckpt} (best combo: {best_run['combo']})")
        else:
            print(f"[warn] Best checkpoint not found at {best_ckpt}")

    print(f"\nGrid search complete. Results saved under {run_root}")


if __name__ == "__main__":
    main()
