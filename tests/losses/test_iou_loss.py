import pytest
import torch

from torchdetect.losses import giou_loss, iou_loss


@pytest.mark.parametrize(
    "pred, target, expected_iou_loss",
    [
        (
            torch.Tensor([[0, 0, 100, 100]]),
            torch.Tensor([[200, 200, 300, 300]]),
            torch.Tensor([1.0]),
        ),
        (
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([0.0]),
        ),
        (
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([[100, 100, 200, 200]]),
            torch.Tensor([0.75]),
        ),
    ],
)
def test_iou(
    pred: torch.Tensor, target: torch.Tensor, expected_iou_loss: torch.Tensor
) -> torch.Tensor:
    assert torch.equal(iou_loss(pred, target), expected_iou_loss)


@pytest.mark.parametrize(
    "pred, target, expected_giou_loss",
    [
        (
            torch.Tensor([[100, 100, 200, 200]]),
            torch.Tensor([[100, 200, 200, 300]]),
            torch.Tensor([1.0]),
        ),
        (
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([0.0]),
        ),
        (
            torch.Tensor([[100, 100, 150, 150]]),
            torch.Tensor([[100, 100, 200, 200]]),
            torch.Tensor([0.75]),
        ),
    ],
)
def test_giou(
    pred: torch.Tensor, target: torch.Tensor, expected_giou_loss: torch.Tensor
) -> torch.Tensor:
    assert torch.equal(giou_loss(pred, target), expected_giou_loss)
