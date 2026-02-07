from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from geovision_ai.models import build_models, classification_loss, segmentation_loss
from geovision_ai.utils.config import TrainConfig


class SegmentationDataset(Dataset):
    def __init__(self, tile_dir: Path, mask_dir: Path):
        self.tiles = sorted(tile_dir.rglob("*.npy"))
        self.mask_dir = mask_dir
        if not self.tiles:
            raise FileNotFoundError(f"No tiles found in {tile_dir}")

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tile_path = self.tiles[idx]
        mask_path = self.mask_dir / tile_path.relative_to(tile_path.parents[1])
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {tile_path} at {mask_path}")
        image = np.load(tile_path).astype(np.float32) / 255.0
        mask = np.load(mask_path).astype(np.int64)
        return torch.tensor(image), torch.tensor(mask)


class ClassificationDataset(Dataset):
    def __init__(self, tile_dir: Path, labels_csv: Path):
        self.tiles = sorted(tile_dir.rglob("*.npy"))
        self.labels = self._load_labels(labels_csv)
        if not self.tiles:
            raise FileNotFoundError(f"No tiles found in {tile_dir}")

    @staticmethod
    def _load_labels(path: Path) -> Dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(f"Missing labels CSV: {path}")
        labels: Dict[str, int] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                labels[row["tile"]] = int(row["label"])
        return labels

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tile_path = self.tiles[idx]
        key = str(tile_path.relative_to(tile_path.parents[1]))
        if key not in self.labels:
            raise KeyError(f"Missing label for {key}")
        image = np.load(tile_path).astype(np.float32) / 255.0
        label = self.labels[key]
        return torch.tensor(image), torch.tensor(label)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="GeoVision AI training")
    parser.add_argument("--data", dest="data_dir", type=Path, required=True)
    parser.add_argument("--output", dest="output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def train_segmentation(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc="Segmentation"):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = segmentation_loss(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def train_classification(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Classification"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = classification_loss(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    models = build_models()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    models.segmentation.to(device)
    models.classification.to(device)

    tiles_dir = config.data_dir / "tiles"
    masks_dir = config.data_dir / "masks"
    labels_csv = config.data_dir / "labels.csv"

    seg_dataset = SegmentationDataset(tiles_dir, masks_dir)
    cls_dataset = ClassificationDataset(tiles_dir, labels_csv)

    seg_loader = DataLoader(seg_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    cls_loader = DataLoader(cls_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    seg_optimizer = torch.optim.AdamW(models.segmentation.parameters(), lr=config.lr)
    cls_optimizer = torch.optim.AdamW(models.classification.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        seg_loss = train_segmentation(models.segmentation, seg_loader, seg_optimizer, str(device))
        cls_loss = train_classification(models.classification, cls_loader, cls_optimizer, str(device))
        print(f"Epoch {epoch + 1}/{config.epochs} - seg_loss={seg_loss:.4f} cls_loss={cls_loss:.4f}")

    torch.save(models.segmentation.state_dict(), config.output_dir / "segmentation.pt")
    torch.save(models.classification.state_dict(), config.output_dir / "classification.pt")
    print(f"Saved models to {config.output_dir}")


if __name__ == "__main__":
    main()
