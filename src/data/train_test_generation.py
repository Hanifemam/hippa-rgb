"""
Materialize train/val/test directly (no JSON files kept).
- Scans DATA_DIR (angles 0/90/180/270 only).
- Splits by apple ID (not individual files) to avoid leakage.
- Copies into dataset/{train,val,test}/class_*.
"""

import os
import shutil
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import read_env_var


def organize_dataset_by_batch_class_round(base_dir: str):
    data_groups = defaultdict(list)
    folders = [f for f in os.listdir(base_dir) if f != "HYP_white_ref" and os.path.isdir(os.path.join(base_dir, f))]

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        for data_type in ["HYP", "RGB", "SPT"]:
            data_type_path = os.path.join(folder_path, data_type)
            if not os.path.exists(data_type_path):
                continue
            for sample_file in os.listdir(data_type_path):
                if not sample_file.startswith(data_type):
                    continue
                parts = sample_file.split("_")
                if len(parts) < 4:
                    continue
                batch = int(parts[1][:2])
                class_id = int(parts[1][2])
                apple_id = parts[1][3:]
                date = parts[2]
                angle = parts[3].split(".")[0]
                if angle not in ["000", "090", "180", "270"]:
                    continue
                ext = parts[3].split(".")[-1].lower()
                if ext not in ["jpg", "jpeg", "png", "tif", "tiff", "spt", "sig"]:
                    continue
                sample_path = os.path.join(data_type_path, sample_file)
                key = (batch, class_id, date, data_type)
                data_groups[key].append(
                    {"path": sample_path, "angle": angle, "apple_id": apple_id, "batch": batch, "class": class_id}
                )

    batch_class_experiments = defaultdict(list)
    for (batch, class_id, date, data_type), samples in data_groups.items():
        key = (batch, class_id, data_type)
        if date not in [exp[0] for exp in batch_class_experiments[key]]:
            batch_class_experiments[key].append((date, samples))
    for key in batch_class_experiments:
        batch_class_experiments[key].sort(key=lambda x: datetime.strptime(x[0], "%y%m%d"))

    organized = {}
    order = {"RGB": 0, "HYP": 1, "SPT": 2}
    for (batch, class_id, data_type), date_samples in batch_class_experiments.items():
        for round_num, (date, samples) in enumerate(date_samples, 1):
            dataset_key = f"batch{batch:02d}_class{class_id}_round{round_num}_{data_type}"
            organized[dataset_key] = {
                "train": [s["path"] for s in samples],
                "batch": batch,
                "class": class_id,
                "round": round_num,
                "date": date,
                "data_type": data_type,
                "sample_count": len(samples),
            }
    organized = dict(
        sorted(
            organized.items(),
            key=lambda item: (item[1]["batch"], order.get(item[1]["data_type"], 3), item[1]["class"], item[1]["round"]),
        )
    )
    return organized


def _apple_id(path: str) -> str:
    name = Path(path).stem
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else parts[0]


def _treatment_from_id(apple_id: str, default: int | None = None) -> int:
    """Extract treatment digit from apple_id; fallback to provided default or -1."""
    return int(apple_id[2]) if len(apple_id) > 2 and apple_id[2].isdigit() else (default if default is not None else -1)


def _allocate_counts(n: int, val_frac: float = 0.15, test_frac: float = 0.15) -> tuple[int, int, int]:
    """Deterministically compute train/val/test counts for a group of size n."""
    if n == 0:
        return 0, 0, 0
    val = max(0, round(n * val_frac))
    test = max(0, round(n * test_frac))
    if n >= 3:
        val = max(1, val)
        test = max(1, test)
    overflow = val + test - n
    while overflow > 0:
        if val >= test and val > 0:
            val -= 1
        elif test > 0:
            test -= 1
        overflow -= 1
    train = n - val - test
    if train == 0 and n > 0:
        if val > test and val > 1:
            val -= 1
        elif test > 1:
            test -= 1
        train = 1
    return train, val, test


def split_dataset(organized_data, output_dir, included_batches=None, exclude_samples=None, modalities=None):
    data = organized_data
    included_batches = set(included_batches or [])
    exclude_samples = set(exclude_samples or [])
    modalities = set(modalities or ["RGB", "SPT", "HYP"])

    filtered = {}
    for key, info in data.items():
        if (not included_batches or info["batch"] in included_batches) and info["data_type"] in modalities:
            keep = [p for p in info["train"] if _apple_id(p) not in exclude_samples]
            if keep:
                sample_id = _apple_id(keep[0])
                if len(sample_id) >= 3 and sample_id[2].isdigit():
                    cls = int(sample_id[2])
                else:
                    cls = info["class"]
                filtered[key] = {**info, "train": keep, "class": cls}

    unique_classes = {info["class"] for info in filtered.values()}
    for split in ["train", "val", "test"]:
        for c in unique_classes:
            Path(output_dir, split, f"class_{c}").mkdir(parents=True, exist_ok=True)

    apple_ids = sorted({_apple_id(p) for info in filtered.values() for p in info["train"]})

    # Stratify by treatment to balance val/test across treatments
    treatment_groups = defaultdict(list)
    for aid in apple_ids:
        treatment_groups[_treatment_from_id(aid)].append(aid)

    train_ids, val_ids, test_ids = [], [], []
    for treatment, ids in sorted(treatment_groups.items()):
        rng = random.Random(123 + treatment)
        ids_copy = ids.copy()
        rng.shuffle(ids_copy)
        train_n, val_n, test_n = _allocate_counts(len(ids_copy), val_frac=0.15, test_frac=0.15)
        val_ids.extend(ids_copy[:val_n])
        test_ids.extend(ids_copy[val_n : val_n + test_n])
        train_ids.extend(ids_copy[val_n + test_n :])

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

    counts = {c: {"total": 0, "train": 0, "val": 0, "test": 0} for c in unique_classes}
    treatment_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})
    for info in filtered.values():
        c = info["class"]
        for path in info["train"]:
            aid = _apple_id(path)
            treatment = _treatment_from_id(aid, info["class"])
            split = "train" if aid in train_set else "val" if aid in val_set else "test"
            counts[c]["total"] += 1
            counts[c][split] += 1
            treatment_counts[treatment]["total"] += 1
            treatment_counts[treatment][split] += 1
            dst = Path(output_dir, split, f"class_{c}", Path(path).name)
            shutil.copy2(path, dst)

    print(f"Split complete at {output_dir}")
    for c in sorted(unique_classes):
        ct = counts[c]
        print(f"class {c}: {ct['train']} train / {ct['val']} val / {ct['test']} test (total {ct['total']})")
    print("Treatment distribution (per split):")
    for t in sorted(treatment_counts):
        ct = treatment_counts[t]
        print(f"  treatment {t}: {ct['train']} train / {ct['val']} val / {ct['test']} test (total {ct['total']})")
    print("Split is deterministic (seed=123); rerun will reproduce the same partitions.")


def main():
    env = read_env_var.evn_var_dict
    root = Path(env["PROJ_NAME"]) / env["TASK"]
    raw = Path(env["DATA_DIR"])
    organized = organize_dataset_by_batch_class_round(str(raw))
    split_dataset(
        organized_data=organized,
        output_dir=root / "dataset",
        included_batches=[4, 7, 8, 10, 12, 13, 23, 32, 34],
        exclude_samples=["10201", "10202", "10601", "10603", "10716"],
        modalities=["RGB"],
    )


if __name__ == "__main__":
    main()
