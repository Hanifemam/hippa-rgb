
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_FILTERS: list[tuple[int, set[int]]] = [
    (7, {2}),
    (23, {7}),
    (32, {3}),
    (34, {2}),
]

def _batch_treat(name: str):
    parts = Path(name).stem.split("_")
    ident = parts[1] if len(parts) > 1 else ""
    return (int(ident[:2]), int(ident[2])) if len(ident) >= 3 and ident[:3].isdigit() else (None, None)


def _match(name: str, filters: Iterable[tuple[int, set[int]]]):
    b, t = _batch_treat(name)
    if b is None or t is None:
        return None
    for fb, ts in filters:
        if b == fb and (not ts or t in ts):
            return b, t
    return None


def match_filter(name: str, filters: Iterable[tuple[int, set[int]]] = DEFAULT_FILTERS):
    return _match(name, filters)


def _prune_dir(root: Path, filters):
    removed, per_pair = [], defaultdict(int)
    if root.exists():
        for path in (p for p in root.rglob("*") if p.is_file()):
            match = _match(path.name, filters)
            if match:
                path.unlink()
                removed.append(path)
                per_pair[match] += 1
    return removed, dict(per_pair)


def filter_by_treatment(filters, data_dir: Path):
    counts, per_pair_total = {}, defaultdict(int)
    for split in ["train", "val", "test"]:
        removed, per_pair = _prune_dir(data_dir / split, filters)
        counts[split] = len(removed)
        for k, v in per_pair.items():
            per_pair_total[k] += v
    return counts, dict(per_pair_total)


def should_drop(name: str, filters: Iterable[tuple[int, set[int]]] = DEFAULT_FILTERS) -> bool:
    return _match(name, filters) is not None


def _read_env(path: Path):
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main():
    root = Path(__file__).resolve().parents[2]
    env = _read_env(root / "env_var.txt")
    candidates = [
        Path(env.get("PROJ_NAME", "")) / env.get("TASK", "") / "dataset",
        Path(env.get("PROJ_NAME", "")) / env.get("TASK", ""),
        Path(env.get("DATA_DIR", "")) / env.get("TASK", ""),
        Path(env.get("DATA_DIR", "")),
        root / "dataset",
        root / "data",
    ]
    data_dir = next((p for p in candidates if (p / "train").exists()), candidates[-1])
    if not (data_dir / "train").exists():
        print(f"No train/val/test under candidates; using {data_dir}")

    counts, per_pair = filter_by_treatment(DEFAULT_FILTERS, data_dir)
    print(f"Pruned splits under {data_dir.resolve()}:")
    for split, n in counts.items():
        print(f"  {split}: removed {n} file(s)")
    if per_pair:
        print("Discarded samples per (batch, treatment):")
        for (b, t), n in sorted(per_pair.items()):
            print(f"  batch {b:02d}, treatment {t}: {n} sample(s)")


if __name__ == "__main__":
    main()
