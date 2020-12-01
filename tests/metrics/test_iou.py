import pytest
import torch

from torchdetect.metrics import giou, iou


@pytest.mark.parametrize(
    "pred, target, expected_iou",
    [
        (
            torch.Tensor([[0, 0, 100, 100]]),
            torch.Tensor([[200, 200, 300, 300]]),
            torch.Tensor([0.0]),
        ),
        (
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([1.0]),
        ),
        (
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([[100, 100, 200, 200]]),
            torch.Tensor([0.25]),
        ),
        (
            torch.Tensor([[100, 100, 150, 200]]),
            torch.Tensor([[100, 100, 200, 200]]),
            torch.Tensor([0.5]),
        ),
    ],
)
def test_iou(
    pred: torch.Tensor, target: torch.Tensor, expected_iou: torch.Tensor
) -> torch.Tensor:
    assert torch.allclose(iou(pred, target), expected_iou)


@pytest.mark.parametrize(
    "pred, target, expected_giou",
    [
        (
            torch.Tensor([[100, 100, 200, 200]]),
            torch.Tensor([[100, 200, 200, 300]]),
            torch.Tensor([0.0]),
        ),
        (
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([1.0]),
        ),
        (
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([[100, 100, 200, 200]]),
            torch.Tensor([0.25]),
        ),
    ],
)
def test_giou(
    pred: torch.Tensor, target: torch.Tensor, expected_giou: torch.Tensor
) -> torch.Tensor:
    assert torch.equal(giou(pred, target), expected_giou)
