from typing import Literal, Dict, Any, Optional, Tuple, List, Type
import io
import os, json, datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

try:
    from result_save import (
        save_results as legacy_save_results,
        save_model_info as legacy_save_model_info,
        save_epoch_results as legacy_save_epoch_results,
    )
except Exception:
    legacy_save_results = None
    legacy_save_model_info = None
    legacy_save_epoch_results = None

# Optional dependencies for metrics and plots
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
except Exception:
    confusion_matrix = None
    ConfusionMatrixDisplay = None

# Headless-safe plotting
try:
    import matplotlib
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =============================
# Robust ResultSaver (always writes locally; also tries result_save.py)
# =============================
class ResultSaver:
    """
    Uses result_save.py if available; ALWAYS writes to base_dir/run_name as well.
    Files:
      - full_model_summary.txt
      - late_fusion_head_summary.txt
      - metrics_epoch.json
      - learning_curves.png
      - confusion_{train,val,test}.png
      - {train,val,test}_results.json
      - best_late_fusion.pth
    """
    def __init__(self, base_dir: str | None = None, run_name: str | None = None):
        # try user's result_save.py
        try:
            import result_save  # noqa
            self._mod = result_save  # type: ignore[attr-defined]
        except Exception:
            self._mod = None

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_dir = Path(base_dir or "results_fusion") / (run_name or ts)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ResultSaver] Writing to: {self.base_dir.resolve()}")

    def _safe_call(self, name: str, *args, **kwargs):
        if self._mod is None:
            return None
        fn = getattr(self._mod, name, None)
        if fn is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    def _write_text(self, filename: str, text: str):
        p = self.base_dir / filename
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[ResultSaver] Wrote: {p}")

    def _write_json(self, filename: str, data):
        p = self.base_dir / filename
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[ResultSaver] Wrote: {p}")

    def _write_fig(self, filename: str, fig):
        if fig is None:
            return
        p = self.base_dir / filename
        try:
            fig.savefig(p, bbox_inches="tight")
            print(f"[ResultSaver] Wrote: {p}")
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

    def _write_checkpoint(self, filename: str, state):
        p = self.base_dir / filename
        try:
            import torch
            torch.save(state, p)
            print(f"[ResultSaver] Wrote: {p}")
        except Exception:
            pass

    # Public API used by fit_with_saving
    def save_text(self, title: str, text: str):
        self._safe_call("save_text", title, text) or self._safe_call("save_model_summary", title, text)
        self._write_text(f"{title.replace(' ', '_').lower()}.txt", text)

    def save_metrics(self, split: str, history: dict):
        self._safe_call("save_metrics", split, history) or self._safe_call("save_learning_curve", history)
        self._write_json("metrics_epoch.json", history)

    def save_figure(self, title: str, fig):
        self._safe_call("save_figure", title, fig) or self._safe_call("save_plot", title, fig)
        fname = f"{title.replace(' ', '_').lower()}.png"
        self._write_fig(fname, fig)

    def save_confusion(self, split: str, fig):
        self._safe_call("save_confusion_matrix", split, fig)
        self._write_fig(f"confusion_{split}.png", fig)

    def save_split_results(self, split: str, metrics: dict):
        self._safe_call("save_split_results", split, metrics)
        self._write_json(f"{split}_results.json", metrics)

    def save_checkpoint(self, state: dict, filename: str = "checkpoint.pth"):
        self._safe_call("save_checkpoint", state, filename) or self._safe_call("save_model", state, filename)
        self._write_checkpoint("best_late_fusion.pth", state)


# =============================
# Fusion Modules
# =============================
FusionMode = Literal["concat", "sum", "prod", "concat+sum+prod", "film"]


class AppleTypeEmbedding(nn.Module):
    """Embedding + normalization for apple type IDs."""
    def __init__(self, num_types: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_types, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, type_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embed(type_ids))


class ProgressionEmbedding(AppleTypeEmbedding):
    """Embedding + normalization for progression stage IDs."""
    pass


class FusionHead(nn.Module):
    """Late fusion head: supports concat, sum, prod, or all three combined."""
    def __init__(self, feat_dim: int, num_classes: int,
                 mode: FusionMode = "concat+sum+prod", p_drop: float = 0.2):
        super().__init__()
        self.mode = mode

        if mode == "concat":
            in_dim = feat_dim * 2
        elif mode in ("sum", "prod"):
            in_dim = feat_dim
        elif mode == "concat+sum+prod":
            in_dim = feat_dim * 3
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, f_img: torch.Tensor, f_type: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            x = torch.cat([f_img, f_type], dim=1)
        elif self.mode == "sum":
            x = f_img + f_type
        elif self.mode == "prod":
            x = f_img * f_type
        else:  # concat+sum+prod
            x = torch.cat([f_img, f_img + f_type, f_img * f_type], dim=1)
        return self.classifier(x)


class FiLMFusionHead(nn.Module):
    """Feature-wise Linear Modulation fusion: apple-type features modulate image features."""
    def __init__(self, feat_dim: int, num_classes: int,
                 hidden_dim: Optional[int] = None, p_drop: float = 0.2):
        super().__init__()
        hidden_dim = hidden_dim or feat_dim
        # FiLM generator adapted from https://github.com/ethanjperez/film
        self.generator = nn.Sequential(
            weight_norm(nn.Linear(feat_dim, hidden_dim)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Linear(hidden_dim, 2 * feat_dim)),
        )
        self.film = FiLMLayer(feat_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, f_img: torch.Tensor, f_type: torch.Tensor) -> torch.Tensor:
        gammas_betas = self.generator(f_type)
        gammas, betas = torch.chunk(gammas_betas, chunks=2, dim=1)
        modulated = self.film(f_img, gammas, betas)
        return self.classifier(modulated)


class FiLMLayer(nn.Module):
    """
    FiLM layer adapted to vector features: applies per-feature gamma/beta.
    Mirrors the affine modulation used in https://github.com/ethanjperez/film.
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.register_parameter("gamma_bias", nn.Parameter(torch.ones(feat_dim)))
        self.register_parameter("beta_bias", nn.Parameter(torch.zeros(feat_dim)))

    def forward(self, features: torch.Tensor, gammas: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        gammas = gammas + self.gamma_bias
        betas = betas + self.beta_bias
        return gammas * features + betas


class ImageProjector(nn.Module):
    """Optional small projector on image features for co-adaptation."""
    def __init__(self, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LateFusionClassifier(nn.Module):
    """Combine image and auxiliary embedding with flexible fusion."""
    def __init__(self, feat_dim: int, num_classes: int,
                 num_types: int, fusion_mode: FusionMode = "concat+sum+prod",
                 film_hidden_dim: Optional[int] = None,
                 embedding_cls: Type[nn.Module] = AppleTypeEmbedding):
        super().__init__()
        self.img_proj = ImageProjector(feat_dim)
        self.type_emb = embedding_cls(num_types=num_types, embed_dim=feat_dim)
        if fusion_mode == "film":
            self.fusion = FiLMFusionHead(
                feat_dim=feat_dim,
                num_classes=num_classes,
                hidden_dim=film_hidden_dim,
            )
        else:
            self.fusion = FusionHead(feat_dim=feat_dim, num_classes=num_classes, mode=fusion_mode)
        self.fusion_mode = fusion_mode

    def forward(self, f_img: torch.Tensor, apple_type_ids: torch.Tensor) -> torch.Tensor:
        f_img = self.img_proj(f_img)
        f_typ = self.type_emb(apple_type_ids)
        return self.fusion(f_img, f_typ)


# =============================
# Summaries
# =============================
def model_summary_string(model: nn.Module, input_shape: Tuple[int, ...]) -> str:
    buf = io.StringIO()
    try:
        from torchinfo import summary
        s = summary(model, input_size=(1,) + input_shape, depth=4, verbose=0)
        buf.write(str(s))
    except Exception:
        buf.write(repr(model))
    return buf.getvalue()


class _FullFusionModel(nn.Module):
    """Wraps backbone + head so we can get a single end-to-end summary."""
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images: torch.Tensor, type_ids: torch.Tensor):
        feats = self.backbone(images)
        return self.head(feats, type_ids)


class _LegacyFusionWrapper(nn.Module):
    """
    Helper so ``result_save.save_model_info`` can introspect backbone/head layout.
    """
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = head

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Legacy wrapper is only for serialization utilities.")


def full_model_summary_string(backbone: nn.Module,
                              head: nn.Module,
                              image_shape: Tuple[int, int, int] = (3, 224, 224)) -> str:
    buf = io.StringIO()
    try:
        from torchinfo import summary
        full = _FullFusionModel(backbone, head)
        s = summary(full,
                    input_size=[(1,) + image_shape, (1,)],  # (images), (apple_type_ids)
                    dtypes=[torch.float32, torch.long],
                    depth=4, verbose=0)
        buf.write(str(s))
    except Exception:
        buf.write(repr(backbone))
        buf.write("\n---\n")
        buf.write(repr(head))
    return buf.getvalue()


# =============================
# Training utilities (with tqdm + AMP)
# =============================
_scaler = GradScaler(enabled=torch.cuda.is_available())

def train_one_epoch(backbone, head, loader, optimizer, device,
                    criterion=nn.CrossEntropyLoss()) -> Dict[str, float]:
    backbone.train()
    head.train()
    total_loss, correct, n = 0.0, 0, 0

    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels, apple_type_ids, _ in loop:
        images = images.to(device)
        labels = labels.to(device)
        apple_type_ids = apple_type_ids.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            feats = backbone(images)
            logits = head(feats, apple_type_ids)
            loss = criterion(logits, labels)

        _scaler.scale(loss).backward()
        _scaler.step(optimizer)
        _scaler.update()

        total_loss += float(loss.detach()) * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
        loop.set_postfix({"loss": f"{total_loss/max(n,1):.4f}", "acc": f"{correct/max(n,1):.4f}"})

    return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1)}


def evaluate(backbone, head, loader, device, compute_conf_mat=True) -> Dict[str, Any]:
    backbone.eval()
    head.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_labels, all_preds = [], []
    criterion = nn.CrossEntropyLoss()

    loop = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, labels, apple_type_ids, _ in loop:
            images = images.to(device)
            labels = labels.to(device)
            apple_type_ids = apple_type_ids.to(device)

            with autocast(enabled=(device.type == "cuda")):
                feats = backbone(images)
                logits = head(feats, apple_type_ids)
                loss = criterion(logits, labels)

            total_loss += float(loss.detach()) * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            n += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            loop.set_postfix({"loss": f"{total_loss/max(n,1):.4f}", "acc": f"{correct/max(n,1):.4f}"})

    metrics = {
        "loss": total_loss / max(n, 1),
        "acc": correct / max(n, 1),
        "labels": all_labels,
        "preds": all_preds,
    }
    if compute_conf_mat and confusion_matrix is not None:
        metrics["confusion_matrix"] = confusion_matrix(all_labels, all_preds)
    return metrics


def plot_learning_curves(history: Dict[str, List[float]]):
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ["train_loss", "val_loss", "train_acc", "val_acc"]:
        if key in history:
            ax.plot(history[key], label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Learning Curves")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(cm: np.ndarray, class_names: Optional[List[str]] = None):
    if plt is None or ConfusionMatrixDisplay is None:
        return None
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, values_format='d', colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def fit_with_saving(backbone: nn.Module,
                    head: LateFusionClassifier,
                    train_loader,
                    val_loader,
                    test_loader,
                    device: torch.device,
                    epochs: int,
                    optimizer: torch.optim.Optimizer,
                    class_names: Optional[List[str]] = None,
                    input_shape: Optional[Tuple[int, ...]] = None,  # head-only summary
                    image_shape: Tuple[int, int, int] = (3, 224, 224),  # full model summary
                    saver: Optional[ResultSaver] = None):
    """
    Full training loop with tqdm progress bars and robust saving.
    """
    saver = saver or ResultSaver()

    # Head-only summary (keep for completeness)
    if input_shape is not None:
        summary_str = model_summary_string(head, input_shape)
        saver.save_text("late_fusion_head_summary", summary_str)

    # Full model summary (backbone + head)
    full_str = full_model_summary_string(backbone, head, image_shape=image_shape)
    saver.save_text("full_model_summary", full_str)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_metrics = train_one_epoch(backbone, head, train_loader, optimizer, device)
        val_metrics = evaluate(backbone, head, val_loader, device, compute_conf_mat=False)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        saver.save_metrics("epoch", history)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                "epoch": epoch,
                "backbone": getattr(backbone, "state_dict")() if hasattr(backbone, "state_dict") else None,
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
            }
            saver.save_checkpoint(best_state, filename="best_late_fusion.pth")

    fig = plot_learning_curves(history)
    if fig is not None:
        saver.save_figure("learning_curves", fig)

    splits = {
        "train": evaluate(backbone, head, train_loader, device, compute_conf_mat=True),
        "val":   evaluate(backbone, head, val_loader, device, compute_conf_mat=True),
        "test":  evaluate(backbone, head, test_loader, device, compute_conf_mat=True),
    }

    # Optional legacy saves (mirrors original apple-type late fusion outputs)
    save_dir = str(saver.base_dir)
    if legacy_save_results is not None:
        try:
            test_metrics = splits["test"]
            legacy_save_results(
                save_dir,
                history.get("train_loss", []),
                history.get("train_acc", []),
                history.get("val_loss", []),
                history.get("val_acc", []),
                test_metrics.get("labels", []),
                test_metrics.get("preds", []),
                class_names or [],
            )
        except Exception as exc:
            print(f"[ResultSaver] save_results failed: {exc}")
    if legacy_save_epoch_results is not None:
        try:
            legacy_save_epoch_results(
                save_dir,
                history.get("train_loss", []),
                history.get("train_acc", []),
                history.get("val_loss", []),
                history.get("val_acc", []),
            )
        except Exception as exc:
            print(f"[ResultSaver] save_epoch_results failed: {exc}")
    if legacy_save_model_info is not None:
        try:
            wrapper = _LegacyFusionWrapper(backbone, head)
            legacy_save_model_info(wrapper, save_dir)
        except Exception as exc:
            print(f"[ResultSaver] save_model_info failed: {exc}")

    for split_name, metrics in splits.items():
        saver.save_split_results(split_name, {k: v for k, v in metrics.items() if k != "confusion_matrix"})
        if "confusion_matrix" in metrics and metrics["confusion_matrix"] is not None:
            fig = plot_confusion(metrics["confusion_matrix"], class_names)
            if fig is not None:
                saver.save_confusion(split_name, fig)

    return {"history": history, "best_val_loss": best_val_loss, "splits": splits}
