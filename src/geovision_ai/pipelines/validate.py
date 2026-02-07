from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from geovision_ai.metrics import compute_segmentation_metrics
from geovision_ai.models import ResUNet
from geovision_ai.pipelines.train import SegmentationDataset
from geovision_ai.utils.config import TrainConfig


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="GeoVision AI validation")
    parser.add_argument("--data", dest="data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    config = TrainConfig(**{k: v for k, v in vars(args).items() if k in TrainConfig.model_fields})
    config.output_dir = Path("artifacts")
    return config


def main() -> None:
    config = parse_args()
    tiles_dir = config.data_dir / "tiles"
    masks_dir = config.data_dir / "masks"
    dataset = SegmentationDataset(tiles_dir, masks_dir)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = ResUNet()
    model.load_state_dict(torch.load(config.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    precision_scores = []
    iou_scores = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for pred, mask in zip(preds, masks.numpy()):
                metrics = compute_segmentation_metrics(pred, mask)
                precision_scores.append(metrics.precision)
                iou_scores.append(metrics.iou)

    avg_precision = float(np.mean(precision_scores))
    avg_iou = float(np.mean(iou_scores))
    print(f"Validation precision: {avg_precision:.4f}, IoU: {avg_iou:.4f}")


if __name__ == "__main__":
    main()
