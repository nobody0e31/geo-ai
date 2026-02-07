from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from geovision_ai.geojson import export_geojson, mask_to_polygons, write_metrics
from geovision_ai.models import EfficientNetClassifier, ResUNet
from geovision_ai.utils.config import InferenceConfig
from geovision_ai.utils.io import iter_raster_paths, iter_tiles

SEGMENT_CLASSES = {
    1: "buildings",
    2: "roads",
    3: "water",
}


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(description="GeoVision AI inference")
    parser.add_argument("--input", dest="input_dir", type=Path, required=True)
    parser.add_argument("--checkpoint-seg", dest="checkpoint_seg", type=Path, required=True)
    parser.add_argument("--checkpoint-cls", dest="checkpoint_cls", type=Path, required=True)
    parser.add_argument("--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()
    return InferenceConfig(**vars(args))


def infer_segmentation(model: ResUNet, tile: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(tile, dtype=torch.float32).unsqueeze(0) / 255.0
        tensor = tensor.to(device)
        logits = model(tensor)
        preds = torch.argmax(logits, dim=1)
    return preds.squeeze(0).cpu().numpy()


def infer_classification(model: EfficientNetClassifier, tile: np.ndarray, device: torch.device) -> tuple[int, float]:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(tile, dtype=torch.float32).unsqueeze(0) / 255.0
        tensor = tensor.to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return int(pred.item()), float(conf.item())


def main() -> None:
    config = parse_args()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    seg_model = ResUNet()
    seg_model.load_state_dict(torch.load(config.checkpoint_seg, map_location=device))
    seg_model.to(device)

    cls_model = EfficientNetClassifier()
    cls_model.load_state_dict(torch.load(config.checkpoint_cls, map_location=device))
    cls_model.to(device)

    metrics_summary: Dict[str, List[float]] = {"tile_confidence": []}

    for path in iter_raster_paths(config.input_dir):
        polygons_by_class: Dict[str, List] = {label: [] for label in SEGMENT_CLASSES.values()}
        tile_confidences = []
        for tile in iter_tiles(path, tile_size=1024, stride=768):
            mask = infer_segmentation(seg_model, tile.image, device)
            for class_id, label in SEGMENT_CLASSES.items():
                class_mask = (mask == class_id).astype(np.uint8)
                polygons = mask_to_polygons(class_mask, tile.transform)
                polygons_by_class[label].extend(polygons)
            _, confidence = infer_classification(cls_model, tile.image, device)
            tile_confidences.append(confidence)

        output_path = config.output_dir / f"{path.stem}.geojson"
        export_geojson(polygons_by_class, tile.crs.to_string() if tile.crs else None, output_path)
        metrics_summary["tile_confidence"].append(float(np.mean(tile_confidences)))

    metrics_path = config.output_dir / "metrics.json"
    write_metrics(metrics_summary, metrics_path)
    print(f"Saved GeoJSON outputs to {config.output_dir}")


if __name__ == "__main__":
    main()
