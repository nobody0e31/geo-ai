from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    raw_dir: Path = Field(default=Path("data/raw"))
    processed_dir: Path = Field(default=Path("data/processed"))
    tile_size: int = 1024
    stride: int = 768


class TrainConfig(BaseModel):
    data_dir: Path = Field(default=Path("data/processed"))
    output_dir: Path = Field(default=Path("artifacts/models"))
    epochs: int = 50
    batch_size: int = 4
    lr: float = 1e-4
    device: Literal["cuda", "cpu"] = "cuda"


class InferenceConfig(BaseModel):
    input_dir: Path = Field(default=Path("data/raw"))
    checkpoint_seg: Path = Field(default=Path("artifacts/models/segmentation.pt"))
    checkpoint_cls: Path = Field(default=Path("artifacts/models/classification.pt"))
    output_dir: Path = Field(default=Path("artifacts/geojson"))
    device: Literal["cuda", "cpu"] = "cuda"
    score_threshold: float = 0.5


class DashboardConfig(BaseModel):
    geojson_dir: Path = Field(default=Path("artifacts/geojson"))
    metrics_path: Path = Field(default=Path("artifacts/metrics.json"))
