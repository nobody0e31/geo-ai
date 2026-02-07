from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def mask_to_polygons(mask: np.ndarray, transform: rasterio.Affine) -> List[Dict]:
    polygons = []
    for geom, value in shapes(mask.astype(np.uint8), transform=transform):
        if value == 0:
            continue
        polygons.append(shape(geom))
    return polygons


def export_geojson(
    polygons_by_class: Dict[str, Iterable],
    crs: str | None,
    output_path: Path,
) -> Path:
    features = []
    for label, polygons in polygons_by_class.items():
        for poly in polygons:
            features.append({"geometry": poly, "label": label})
    gdf = gpd.GeoDataFrame(features, geometry="geometry", crs=crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    return output_path


def write_metrics(metrics: Dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return output_path
