from __future__ import annotations

import argparse
from pathlib import Path

from geovision_ai.preprocessing import preprocess_dataset
from geovision_ai.utils.config import DataConfig


def parse_args() -> DataConfig:
    parser = argparse.ArgumentParser(description="GeoVision AI preprocessing")
    parser.add_argument("--input", dest="raw_dir", type=Path, required=True)
    parser.add_argument("--output", dest="processed_dir", type=Path, required=True)
    parser.add_argument("--tile-size", dest="tile_size", type=int, default=1024)
    parser.add_argument("--stride", dest="stride", type=int, default=768)
    args = parser.parse_args()
    return DataConfig(**vars(args))


def main() -> None:
    config = parse_args()
    images, tiles = preprocess_dataset(
        config.raw_dir,
        config.processed_dir,
        config.tile_size,
        config.stride,
    )
    print(f"Processed {images} images into {tiles} tiles.")


if __name__ == "__main__":
    main()
