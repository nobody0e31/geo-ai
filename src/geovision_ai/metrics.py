from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationMetrics:
    precision: float
    iou: float


@dataclass
class ClassificationMetrics:
    precision: float


def compute_segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> SegmentationMetrics:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = (pred & true).sum()
    union = (pred | true).sum()
    precision = intersection / max(pred.sum(), 1)
    iou = intersection / max(union, 1)
    return SegmentationMetrics(precision=float(precision), iou=float(iou))


def compute_classification_precision(preds: np.ndarray, labels: np.ndarray) -> ClassificationMetrics:
    correct = (preds == labels).sum()
    precision = correct / max(len(labels), 1)
    return ClassificationMetrics(precision=float(precision))
