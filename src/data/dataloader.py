# hippa_dataloaders.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def _tf(train: bool, s: int) -> Callable:
    if train:
        return T.Compose([
            T.RandomResizedCrop(s, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(int(s * 256 / 224)),
        T.CenterCrop(s),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

def _imgs(root: Path, split: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in (root / split).rglob("*") if p.is_file() and p.suffix.lower() in exts]

def _enc(series: pd.Series) -> Dict[str, int]:
    vals = sorted({str(v) for v in series.dropna().tolist()})
    return {"__UNK__": 0, **{v: i + 1 for i, v in enumerate(vals)}}

def _id(m: Dict[str, int], v) -> int:
    return m.get(str(v), m.get("__UNK__", 0)) if pd.notna(v) else m.get("__UNK__", 0)

class HIPPASet(Dataset):
    # modes: "img", "img_cult", "img_prog", "img_prog_cult"
    def __init__(
        self, root: Union[str, Path], split: str, mode: str, df: pd.DataFrame,
        y2i: Dict[str, int], c2i: Optional[Dict[str, int]], p2i: Optional[Dict[str, int]],
        tfm: Callable, strict: bool = True, cult_col: str = "cultivar", prog_col: str = "progression",
    ):
        self.mode, self.df, self.y2i, self.c2i, self.p2i, self.tfm = mode, df, y2i, c2i, p2i, tfm
        self.cult_col, self.prog_col = cult_col, prog_col
        self.items = [(p, y2i[p.parent.name], p.name) for p in _imgs(Path(root), split)]
        if not self.items: raise ValueError(f"No images under {(Path(root)/split)}")
        if strict:
            miss = [n for _, _, n in self.items if n not in df.index]
            if miss: raise KeyError(f"CSV missing {len(miss)} image(s), e.g. {miss[:5]}")

    def __len__(self): return len(self.items)

    def __getitem__(self, i: int):
        p, y, name = self.items[i]
        x = self.tfm(Image.open(p).convert("RGB"))
        if self.mode == "img": return x, torch.tensor(y)
        row = self.df.loc[name] if name in self.df.index else None

        if self.mode == "img_cult":
            if self.c2i is None: raise RuntimeError("cultivar encoder missing")
            c = _id(self.c2i, None if row is None else row[self.cult_col])
            return x, torch.tensor(c), torch.tensor(y)

        if self.mode == "img_prog":
            if self.p2i is None: raise RuntimeError("progression encoder missing")
            pr = _id(self.p2i, None if row is None else row[self.prog_col])
            return x, torch.tensor(pr), torch.tensor(y)

        if self.mode == "img_prog_cult":
            if self.p2i is None or self.c2i is None: raise RuntimeError("encoders missing")
            pr = _id(self.p2i, None if row is None else row[self.prog_col])
            c = _id(self.c2i, None if row is None else row[self.cult_col])
            return x, torch.tensor(pr), torch.tensor(c), torch.tensor(y)

        raise ValueError(f"Unknown mode: {self.mode}")

@dataclass
class HIPPADataLoader:
    image_dir: Union[str, Path]
    csv_path: Union[str, Path]  # image_progression_cultivar.csv
    batch_size: int = 32
    num_workers: int = 4
    img_size: int = 224
    pin_memory: bool = True
    drop_last: bool = True
    strict_csv_match: bool = True
    image_col: str = "image_name"
    cult_col: str = "cultivar"
    prog_col: str = "progression"
    train_tf: Optional[Callable] = None
    eval_tf: Optional[Callable] = None

    def __post_init__(self):
        self.image_dir, self.csv_path = Path(self.image_dir), Path(self.csv_path)
        classes = sorted([p.name for p in (self.image_dir / "train").iterdir() if p.is_dir()])
        if not classes: raise ValueError(f"No class folders under {self.image_dir/'train'}")
        self.label2id = {c: i for i, c in enumerate(classes)}

        df = pd.read_csv(self.csv_path)
        for col in [self.image_col, self.cult_col, self.prog_col]:
            if col not in df.columns: raise KeyError(f"CSV missing '{col}'. Found: {list(df.columns)}")
        self.df = df.set_index(self.image_col, drop=False)

        train_names = {p.name for p in _imgs(self.image_dir, "train")}
        df_tr = df[df[self.image_col].astype(str).isin(train_names)]
        if self.strict_csv_match:
            miss = sorted(train_names - set(df_tr[self.image_col].astype(str)))
            if miss: raise KeyError(f"CSV missing {len(miss)} train image(s), e.g. {miss[:5]}")
        self.cultivar2id, self.progression2id = _enc(df_tr[self.cult_col]), _enc(df_tr[self.prog_col])

        self.train_tf = self.train_tf or _tf(True, self.img_size)
        self.eval_tf  = self.eval_tf  or _tf(False, self.img_size)

    def _loader(self, split: str, mode: str):
        ds = HIPPASet(
            root=self.image_dir, split=split, mode=mode, df=self.df, y2i=self.label2id,
            c2i=self.cultivar2id, p2i=self.progression2id,
            tfm=self.train_tf if split == "train" else self.eval_tf,
            strict=self.strict_csv_match, cult_col=self.cult_col, prog_col=self.prog_col,
        )
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=(split == "train"),
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            drop_last=(self.drop_last and split == "train"),
        )

    def dataloader_image(self, split: str = "train"): return self._loader(split, "img")
    def dataloader_image_cultivar(self, split: str = "train"): return self._loader(split, "img_cult")
    def dataloader_image_progression(self, split: str = "train"): return self._loader(split, "img_prog")
    def dataloader_image_progression_cultivar(self, split: str = "train"): return self._loader(split, "img_prog_cult")
