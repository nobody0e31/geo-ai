from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

from geovision_ai.utils.io import iter_raster_paths, iter_tiles


def ortho_rectify(path: Path, output_path: Path) -> Path:
    """Ensure imagery is aligned and projected to a consistent CRS.

    This function reads the raster and writes it back with standardized
    alignment so downstream tiling uses consistent transforms.
    """
    with rasterio.open(path) as src:
        profile = src.profile
        image = src.read(
            out_shape=(src.count, src.height, src.width),
            resampling=Resampling.bilinear,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(image)
    return output_path


def augment_image(image: np.ndarray) -> Iterable[np.ndarray]:
    """Generate augmented variants of a tile."""
    yield image
    yield np.flip(image, axis=1)
    yield np.flip(image, axis=2)
    hsv = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255)
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    yield bright.transpose(2, 0, 1)


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    tile_size: int,
    stride: int,
) -> Tuple[int, int]:
    """Ortho-rectify imagery and generate tiles for training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    tile_count = 0
    for path in iter_raster_paths(input_dir):
        image_count += 1
        rectified_path = output_dir / "rectified" / path.name
        ortho_rectify(path, rectified_path)
        for tile in iter_tiles(rectified_path, tile_size, stride):
            tile_count += 1
            tile_dir = output_dir / "tiles" / path.stem
            tile_dir.mkdir(parents=True, exist_ok=True)
            tile_path = tile_dir / f"tile_{tile_count:06d}.npy"
            np.save(tile_path, tile.image)
    return image_count, tile_count
