__all__ = ["iou_loss", "giou_loss"]

import torch

from ..metrics import giou, iou


def iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes Intersection over Union loss.

    Arguments:
        pred: an N x 4 tensor of prediction bounding boxes
        target: an N x 4 tensor of target bounding boxes

    Returns:
        a N tensor of IoU calculations
    """
    iou_ = iou(pred, target)
    return 1 - iou_


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes Generalized Intersection over Union loss.

    Arguments:
        pred: an N x 4 tensor of prediction bounding boxes
        target: an N x 4 tensor of target bounding boxes

    Returns:
        a N tensor of IoU calculations
    """
    giou_ = giou(pred, target)
    return 1 - giou_
