
"""Lightweight training helpers for image classification."""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch
from tqdm.auto import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Single training epoch."""
    model.train()
    train_loss, train_acc = 0.0, 0.0

    loop = tqdm(dataloader, desc="train_step", leave=False)
    for batch_idx, (X, y) in enumerate(loop):
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        train_loss += loss.item()
        train_acc += (preds == y).float().mean().item()

        loop.set_postfix(loss=f"{train_loss / max(batch_idx + 1, 1):.4f}",
                         acc=f"{train_acc / max(batch_idx + 1, 1):.4f}")

    n_batches = max(len(dataloader), 1)
    return train_loss / n_batches, train_acc / n_batches


def eval_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Single evaluation epoch."""
    model.eval()
    eval_loss, eval_acc = 0.0, 0.0

    with torch.inference_mode():
        loop = tqdm(dataloader, desc="eval_step", leave=False)
        for batch_idx, (X, y) in enumerate(loop):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            preds = logits.argmax(dim=1)
            eval_loss += loss.item()
            eval_acc += (preds == y).float().mean().item()

            loop.set_postfix(loss=f"{eval_loss / max(batch_idx + 1, 1):.4f}",
                             acc=f"{eval_acc / max(batch_idx + 1, 1):.4f}")

    n_batches = max(len(dataloader), 1)
    return eval_loss / n_batches, eval_acc / n_batches


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Train and validate a model, returning epoch metrics."""
    model.to(device)
    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc = eval_step(model, val_dataloader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

    return history
