from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.late_fusion_embeddings import LateFusionCombiner, LateFusionHead
from models.model_builder import Conv4DCNN, build_model
from training.engine import eval_step, train_step
from training.training import _forward_factory


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("concat", torch.tensor([[1.0, 2.0, 3.0, 4.0]])),
        ("sum", torch.tensor([[4.0, 6.0]])),
        ("prod", torch.tensor([[3.0, 8.0]])),
        (
            "concat+sum+prod",
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 4.0, 6.0, 3.0, 8.0]]),
        ),
    ],
)
def test_late_fusion_combiner_modes(mode: str, expected: torch.Tensor) -> None:
    combiner = LateFusionCombiner(feat_dim=2, fusion_mode=mode)
    combiner.img_norm = nn.Identity()

    actual = combiner(
        torch.tensor([[1.0, 2.0]]),
        cultivar_feats=torch.tensor([[3.0, 4.0]]),
    )

    torch.testing.assert_close(actual, expected)


def test_late_fusion_head_supports_both_metadata_inputs() -> None:
    head = LateFusionHead(
        feat_dim=8,
        num_classes=3,
        num_cultivars=4,
        num_progressions=5,
        fusion_mode="concat+sum+prod",
        dropout=0.0,
    )

    logits = head(
        torch.randn(2, 8),
        cultivar_ids=torch.tensor([0, 3]),
        progression_ids=torch.tensor([1, 4]),
    )

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_model_factory_builds_lightweight_model_and_rejects_unknown_name() -> None:
    model = build_model(
        "CONV4DCNN",
        in_channels=3,
        num_classes=4,
        img_size=64,
        hidden_dim=16,
        dropout=0.0,
    )

    assert isinstance(model, Conv4DCNN)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 4)

    with pytest.raises(ValueError, match="Unknown model"):
        build_model("missing-model")


class _MetadataModel(nn.Module):
    def forward(
        self,
        images: torch.Tensor,
        *,
        cultivar_ids: torch.Tensor | None = None,
        progression_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = images
        if cultivar_ids is not None:
            result = result + cultivar_ids.unsqueeze(1)
        if progression_ids is not None:
            result = result + 10 * progression_ids.unsqueeze(1)
        return result


@pytest.mark.parametrize(
    ("mode", "batch", "expected"),
    [
        (
            "img",
            (torch.tensor([[1.0, 2.0]]), torch.tensor([1])),
            torch.tensor([[1.0, 2.0]]),
        ),
        (
            "img_cult",
            (torch.tensor([[1.0, 2.0]]), torch.tensor([3]), torch.tensor([1])),
            torch.tensor([[4.0, 5.0]]),
        ),
        (
            "img_prog",
            (torch.tensor([[1.0, 2.0]]), torch.tensor([3]), torch.tensor([1])),
            torch.tensor([[31.0, 32.0]]),
        ),
        (
            "img_prog_cult",
            (
                torch.tensor([[1.0, 2.0]]),
                torch.tensor([3]),
                torch.tensor([4]),
                torch.tensor([1]),
            ),
            torch.tensor([[35.0, 36.0]]),
        ),
    ],
)
def test_forward_factory_routes_batch_metadata(
    mode: str, batch: tuple[torch.Tensor, ...], expected: torch.Tensor
) -> None:
    logits, labels = _forward_factory(mode)(_MetadataModel(), batch, torch.device("cpu"))

    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(labels, torch.tensor([1]))


def test_training_engine_updates_parameters_and_evaluates() -> None:
    torch.manual_seed(0)
    features = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    original_weight = model.weight.detach().clone()

    train_loss, train_accuracy = train_step(
        model, loader, loss_fn, optimizer, torch.device("cpu")
    )
    eval_loss, eval_accuracy = eval_step(model, loader, loss_fn, torch.device("cpu"))

    assert math.isfinite(train_loss)
    assert math.isfinite(eval_loss)
    assert 0.0 <= train_accuracy <= 1.0
    assert 0.0 <= eval_accuracy <= 1.0
    assert not torch.equal(model.weight.detach(), original_weight)
