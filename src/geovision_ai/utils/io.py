from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass
class RasterTile:
    image: np.ndarray
    transform: rasterio.Affine
    crs: rasterio.crs.CRS | None
    path: Path


def iter_raster_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.tif"):
        yield path
    for path in root.rglob("*.tiff"):
        yield path


def read_raster(path: Path) -> Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS | None]:
    with rasterio.open(path) as src:
        image = src.read(out_dtype="uint8")
        transform = src.transform
        crs = src.crs
    return image, transform, crs


def iter_tiles(
    path: Path,
    tile_size: int,
    stride: int,
) -> Iterator[RasterTile]:
    with rasterio.open(path) as src:
        width = src.width
        height = src.height
        for top in range(0, height, stride):
            for left in range(0, width, stride):
                window = Window(left, top, tile_size, tile_size)
                image = src.read(window=window, boundless=True, fill_value=0)
                transform = src.window_transform(window)
                yield RasterTile(
                    image=image,
                    transform=transform,
                    crs=src.crs,
                    path=path,
                )
