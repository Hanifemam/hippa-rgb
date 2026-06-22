"""Run the fixed apple, progression, and disease comparison experiments."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.dataloader import HIPPADataLoader  # noqa: E402
from models.apple_type_prediction import train_apple_type  # noqa: E402
from models.progression_prediction import train_progression_late_fusion  # noqa: E402
from training.training import _forward_factory, run_single_experiment  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_FUSION_MODES = ["sum"]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _image_names(root: Path) -> set[str]:
    for split in ("train", "val"):
        if not (root / split).exists():
            raise FileNotFoundError(f"Disease split not found: {root / split}")
    names = [
        path.name
        for split in ("train", "val", "test")
        for path in (root / split).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not names:
        raise ValueError(f"No disease images found under {root}")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"Disease image names must be unique, e.g. {sorted(duplicates)[:5]}")
    return set(names)


def _prepare_true_metadata(source: Path, disease_root: Path, output: Path) -> Path:
    frame = pd.read_csv(source)
    required = ["sample_name", "cultivar", "progression"]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"True metadata is missing columns: {missing_columns}")
    if frame["sample_name"].duplicated().any():
        duplicates = frame.loc[frame["sample_name"].duplicated(), "sample_name"].tolist()
        raise ValueError(f"True metadata has duplicate sample names, e.g. {duplicates[:5]}")

    names = _image_names(disease_root)
    frame["sample_name"] = frame["sample_name"].astype(str)
    missing_names = sorted(names - set(frame["sample_name"]))
    if missing_names:
        raise KeyError(f"True metadata is missing {len(missing_names)} disease images, e.g. {missing_names[:5]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[frame["sample_name"].isin(names), required].sort_values("sample_name").to_csv(
        output, index=False
    )
    return output


def _combine_predicted_cultivar_with_true_progression(
    predicted_cultivar_csv: Path, true_metadata_csv: Path, output: Path
) -> Path:
    cultivar = pd.read_csv(predicted_cultivar_csv)[["sample_name", "cultivar"]]
    progression = pd.read_csv(true_metadata_csv)[["sample_name", "progression"]]
    combined = cultivar.merge(progression, on="sample_name", how="inner", validate="one_to_one")
    if len(combined) != len(progression):
        raise ValueError("Predicted cultivar CSV does not cover every disease image.")
    combined.sort_values("sample_name").to_csv(output, index=False)
    return output


def _has_test_images(root: Path) -> bool:
    test_dir = root / "test"
    return test_dir.exists() and any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTS for path in test_dir.rglob("*")
    )


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    history = result.get("history", {})
    return {
        "best_val_loss": result.get("best_val_loss"),
        "best_epoch": result.get("best_epoch"),
        "final_val_accuracy": history.get("val_acc", [None])[-1] if history.get("val_acc") else None,
        "evaluation": result.get("evaluation", {}),
    }


def _run_disease_experiment(
    *,
    name: str,
    mode: str,
    learning_rate: float,
    dropout: float,
    optimizer: str,
    fusion_mode: str | None,
    metadata_csv: Path,
    disease_root: Path,
    output_root: Path,
    device: torch.device,
    cultivar2id: dict[str, int],
    progression2id: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    data = HIPPADataLoader(
        image_dir=disease_root,
        csv_path=metadata_csv,
        batch_size=args.disease_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        img_size=224,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        strict_csv_match=True,
        use_trivial_augment=True,
    )
    data.cultivar2id = dict(cultivar2id)
    data.progression2id = dict(progression2id)
    loader_method = {
        "img": data.dataloader_image,
        "img_cult": data.dataloader_image_cultivar,
        "img_prog": data.dataloader_image_progression,
        "img_prog_cult": data.dataloader_image_progression_cultivar,
    }[mode]
    hparams: dict[str, Any] = {
        "model_name": "hybrid_augmented" if mode == "img" else "hybrid_augmented_latefusion",
        "image_dir": str(disease_root),
        "csv_path": str(metadata_csv),
        "batch_size": args.disease_batch_size,
        "img_size": 224,
        "in_channels": 3,
        "hidden_dim": 512,
        "dropout": dropout,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "learning_rate": learning_rate,
        "epochs": args.disease_epochs,
        "early_stopping_patience": 5,
        "optimizer": optimizer,
        "loss_fn": "CrossEntropyLoss",
        "device": str(device),
        "pretrained": True,
        "class_to_idx": data.label2id,
        "num_classes": len(data.label2id),
    }
    if fusion_mode is not None:
        hparams["fusion_mode"] = fusion_mode
    if "cult" in mode:
        hparams["num_cultivars"] = len(data.cultivar2id)
    if "prog" in mode:
        hparams["num_progressions"] = len(data.progression2id)

    return run_single_experiment(
        output_root / name,
        hparams,
        device,
        loader_method("train"),
        loader_method("val"),
        forward_fn=_forward_factory(mode),
        test_loader=loader_method("test") if _has_test_images(disease_root) else None,
    )


def run(args: argparse.Namespace) -> Path:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_root = Path(args.results) / f"final_experiments_{datetime.now():%Y%m%d-%H%M%S}"
    metadata_root = run_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    disease_root = Path(args.disease_data)
    true_metadata = _prepare_true_metadata(
        Path(args.true_metadata), disease_root, metadata_root / "true_metadata.csv"
    )
    reference_data = HIPPADataLoader(
        image_dir=disease_root,
        csv_path=true_metadata,
        batch_size=args.disease_batch_size,
        num_workers=0,
        seed=args.seed,
        img_size=224,
        pin_memory=False,
        drop_last=False,
        strict_csv_match=True,
        use_trivial_augment=False,
    )
    predicted_cultivar = metadata_root / "predicted_cultivar.csv"
    predicted_progression_true_cultivar = (
        metadata_root / "true_cultivar_predicted_progression.csv"
    )
    predicted_both = metadata_root / "predicted_cultivar_predicted_progression.csv"
    predicted_cultivar_true_progression = (
        metadata_root / "predicted_cultivar_true_progression.csv"
    )
    disease_run_count = (
        len(args.dropouts)
        * len(args.disease_learning_rates)
        * len(args.optimizers)
        * (1 + 6 * len(args.fusion_modes))
    )
    progression_run_count = (
        len(args.progression_resnets)
        * len(args.progression_learning_rates)
        * len(args.progression_weight_decays)
        * len(args.progression_fusion_modes)
    )

    config = {
        "device": str(device),
        "seed": args.seed,
        "disease_data": str(disease_root),
        "apple_data": args.apple_data,
        "progression_data": args.progression_data,
        "true_metadata": str(true_metadata),
        "progression_cultivar_csv": args.progression_cultivar_csv,
        "planned_training_runs": {
            "disease": disease_run_count,
            "apple_type": 1,
            "progression": progression_run_count,
            "total": disease_run_count + progression_run_count + 1,
        },
        "hyperparameter_grid": {
            "disease": {
                "model": "hybrid_augmented",
                "late_fusion_model": "hybrid_augmented_latefusion",
                "fusion_modes": args.fusion_modes,
                "epochs": args.disease_epochs,
                "batch_size": args.disease_batch_size,
                "learning_rates": args.disease_learning_rates,
                "dropouts": args.dropouts,
                "optimizers": args.optimizers,
            },
            "apple_type": {
                "model": "resnet152",
                "epochs": args.apple_epochs,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "unfreeze_layers": 0,
            },
            "progression": {
                "resnets": args.progression_resnets,
                "fusion_modes": args.progression_fusion_modes,
                "epochs": args.progression_epochs,
                "batch_size": 32,
                "learning_rates": args.progression_learning_rates,
                "weight_decays": args.progression_weight_decays,
            },
        },
    }
    manifest: dict[str, Any] = {"config": config, "runs": {}}
    _write_json(run_root / "manifest.json", manifest)

    apple_result = train_apple_type(
        args.apple_data,
        prediction_data=disease_root,
        output_csv=predicted_cultivar,
        epochs=args.apple_epochs,
        batch_size=32,
        num_workers=args.num_workers,
        lr=1e-4,
        weight_decay=1e-4,
        unfreeze_layers=0,
        patience=5,
        device=str(device),
        save_base_dir=run_root / "03_apple_type_prediction",
        seed=args.seed,
    )
    manifest["runs"]["03_apple_type_prediction"] = apple_result
    _write_json(run_root / "manifest.json", manifest)

    progression_grid = list(
        itertools.product(
            args.progression_resnets,
            args.progression_learning_rates,
            args.progression_weight_decays,
            args.progression_fusion_modes,
        )
    )
    progression_candidates = []
    progression_metadata_root = metadata_root / "progression_candidates"
    progression_metadata_root.mkdir(parents=True, exist_ok=True)
    print(f"\nPlanned progression grid runs: {len(progression_grid)}")
    for run_index, (resnet, learning_rate, weight_decay, fusion_mode) in enumerate(
        progression_grid, start=1
    ):
        candidate = (
            f"{resnet}_lr{learning_rate:.0e}_wd{weight_decay:.0e}"
            f"_fusion{fusion_mode.replace('+', '_')}"
        )
        true_output = progression_metadata_root / f"{candidate}_true_cultivar.csv"
        predicted_output = progression_metadata_root / f"{candidate}_predicted_cultivar.csv"
        print(
            f"\n=== Progression grid {run_index}/{len(progression_grid)}: {candidate} ==="
        )
        result = train_progression_late_fusion(
            args.progression_data,
            cultivar_csv=args.progression_cultivar_csv,
            epochs=args.progression_epochs,
            early_stopping_patience=5,
            unfreeze_layers=-1,
            fusion_mode=fusion_mode,
            batch_size=32,
            num_workers=args.num_workers,
            lr=learning_rate,
            weight_decay=weight_decay,
            img_size=224,
            resnet_name=resnet,
            pretrained=True,
            device=str(device),
            save_base_dir=run_root / "04_progression_prediction",
            prediction_data=disease_root,
            prediction_jobs=[
                (true_metadata, true_output),
                (predicted_cultivar, predicted_output),
            ],
            seed=args.seed,
        )
        summary = {
            "candidate": candidate,
            "resnet": resnet,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "fusion_mode": fusion_mode,
            "save_dir": result["save_dir"],
            "best_val_loss": result["best_val_loss"],
            "best_epoch": result["best_epoch"],
            "true_cultivar_csv": str(true_output),
            "predicted_cultivar_csv": str(predicted_output),
        }
        progression_candidates.append(summary)
        manifest["runs"][f"04_progression_prediction_{candidate}"] = summary
        _write_json(run_root / "manifest.json", manifest)

    progression_comparison = pd.DataFrame(progression_candidates).sort_values(
        "best_val_loss"
    )
    progression_comparison.to_csv(
        run_root / "progression_comparison.csv", index=False
    )
    best_progression = min(
        progression_candidates, key=lambda candidate: candidate["best_val_loss"]
    )
    shutil.copy2(
        best_progression["true_cultivar_csv"],
        predicted_progression_true_cultivar,
    )
    shutil.copy2(best_progression["predicted_cultivar_csv"], predicted_both)
    _write_json(run_root / "best_progression_run.json", best_progression)
    manifest["best_progression_run"] = best_progression
    _write_json(run_root / "manifest.json", manifest)

    _combine_predicted_cultivar_with_true_progression(
        predicted_cultivar, true_metadata, predicted_cultivar_true_progression
    )

    conditions = [
        ("01_disease_image_only", "img", true_metadata),
        ("02_disease_true_cultivar_true_progression", "img_prog_cult", true_metadata),
        ("05a_disease_true_cultivar", "img_cult", true_metadata),
        ("05b_disease_predicted_cultivar", "img_cult", predicted_cultivar_true_progression),
        ("06a_disease_true_progression", "img_prog", true_metadata),
        (
            "06b_disease_predicted_progression",
            "img_prog",
            predicted_progression_true_cultivar,
        ),
        ("07_disease_predicted_cultivar_predicted_progression", "img_prog_cult", predicted_both),
    ]
    experiment_grid = []
    for condition, mode, metadata_csv in conditions:
        fusion_modes = [None] if mode == "img" else args.fusion_modes
        for dropout, learning_rate, optimizer, fusion_mode in itertools.product(
            args.dropouts,
            args.disease_learning_rates,
            args.optimizers,
            fusion_modes,
        ):
            suffix = (
                f"drop{dropout:g}_lr{learning_rate:.0e}_opt{optimizer.lower()}"
                + (f"_fusion{fusion_mode.replace('+', '_')}" if fusion_mode else "")
            )
            experiment_grid.append(
                (f"{condition}_{suffix}", condition, mode, metadata_csv, learning_rate,
                 dropout, optimizer, fusion_mode)
            )

    print(f"\nPlanned disease grid runs: {len(experiment_grid)}")
    comparison_rows = []
    for run_index, (
        run_name,
        condition,
        mode,
        metadata_csv,
        learning_rate,
        dropout,
        optimizer,
        fusion_mode,
    ) in enumerate(experiment_grid, start=1):
        print(f"\n=== Disease grid {run_index}/{len(experiment_grid)}: {run_name} ===")
        result = _run_disease_experiment(
            name=run_name,
            mode=mode,
            learning_rate=learning_rate,
            dropout=dropout,
            optimizer=optimizer,
            fusion_mode=fusion_mode,
            metadata_csv=metadata_csv,
            disease_root=disease_root,
            output_root=run_root,
            device=device,
            cultivar2id=reference_data.cultivar2id,
            progression2id=reference_data.progression2id,
            args=args,
        )
        summary = {
            "condition": condition,
            "mode": mode,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "optimizer": optimizer,
            "fusion_mode": fusion_mode,
            **_result_summary(result),
        }
        manifest["runs"][run_name] = summary
        comparison_rows.append({"experiment": run_name, **summary})
        _write_json(run_root / "manifest.json", manifest)

    comparison = pd.json_normalize(comparison_rows, sep="_")
    comparison.to_csv(run_root / "disease_comparison.csv", index=False)
    best_rows = comparison.loc[
        comparison.groupby("condition")["evaluation_val_accuracy"].idxmax()
    ].sort_values("condition")
    best_rows.to_csv(run_root / "best_disease_runs.csv", index=False)
    print(f"\nResults: {run_root}")
    print(f"TensorBoard: tensorboard --logdir {run_root}")
    return run_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease-data", required=True, help="Disease train/val/test root.")
    parser.add_argument("--true-metadata", required=True, help="True disease metadata CSV.")
    parser.add_argument("--apple-data", required=True, help="Apple-type train/val/test root.")
    parser.add_argument("--progression-data", required=True, help="Progression train/val/test root.")
    parser.add_argument(
        "--progression-cultivar-csv",
        required=True,
        help="True cultivar CSV covering progression-training images.",
    )
    parser.add_argument("--results", default="results/final_experiments")
    parser.add_argument("--disease-epochs", type=int, default=20)
    parser.add_argument("--apple-epochs", type=int, default=10)
    parser.add_argument("--progression-epochs", type=int, default=20)
    parser.add_argument(
        "--progression-resnets",
        nargs="+",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        default=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    )
    parser.add_argument(
        "--progression-learning-rates",
        type=float,
        nargs="+",
        default=[3e-4, 1e-4, 3e-5],
    )
    parser.add_argument(
        "--progression-weight-decays",
        type=float,
        nargs="+",
        default=[1e-4],
    )
    parser.add_argument(
        "--progression-fusion-modes",
        nargs="+",
        choices=["concat", "sum", "prod", "concat+sum+prod"],
        default=DEFAULT_FUSION_MODES,
        help="Progression fusion modes to search; default: sum.",
    )
    parser.add_argument("--disease-batch-size", type=int, default=16)
    parser.add_argument(
        "--disease-learning-rates",
        type=float,
        nargs="+",
        default=[3e-4, 1e-4, 3e-5],
    )
    parser.add_argument(
        "--fusion-modes",
        nargs="+",
        choices=["concat", "sum", "prod", "concat+sum+prod"],
        default=DEFAULT_FUSION_MODES,
        help="Disease fusion modes to search; default: sum.",
    )
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.3])
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=["Adam", "AdamW", "SGD"],
        default=["Adam"],
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
