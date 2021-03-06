__all__ = ["iou", "giou"]

import torch


def iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes Intersection over Union.

    Arguments:
        pred: an N x 4 tensor of prediction bounding boxes
        target: an M x 4 tensor of target bounding boxes

    Returns:
        a N x M tensor of iou values between every pair of prediction bounding box and
        target bounding box
    """
    x_min = torch.max(pred[:, None, 0], target[:, 0])
    y_min = torch.max(pred[:, None, 1], target[:, 1])
    x_max = torch.min(pred[:, None, 2], target[:, 2])
    y_max = torch.min(pred[:, None, 3], target[:, 3])
    dx, dy = (x_max - x_min).clamp(min=0), (y_max - y_min).clamp(min=0)
    intersection = dx * dy
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = pred_area[:, None] + target_area - intersection
    return intersection / union


def giou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes Generalized Intersection over Union.

    Arguments:
        pred: an N x 4 tensor of prediction bounding boxes
        target: an M x 4 tensor of target bounding boxes

    Returns:
        a N x M tensor of iou values between every pair of prediction bounding box and
        target bounding box
    """
    x_min = torch.max(pred[:, None, 0], target[:, 0])
    y_min = torch.max(pred[:, None, 1], target[:, 1])
    x_max = torch.min(pred[:, None, 2], target[:, 2])
    y_max = torch.min(pred[:, None, 3], target[:, 3])
    dx, dy = (x_max - x_min).clamp(min=0), (y_max - y_min).clamp(min=0)
    intersection = dx * dy
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = pred_area[:, None] + target_area - intersection
    C_x_min = torch.min(pred[:, None, 0], target[:, 0])
    C_y_min = torch.min(pred[:, None, 1], target[:, 1])
    C_x_max = torch.max(pred[:, None, 2], target[:, 2])
    C_y_max = torch.max(pred[:, None, 3], target[:, 3])
    C_area = (C_x_max - C_x_min) * (C_y_max - C_y_min)
    iou = intersection / union
    return iou - (C_area - union) / C_area
