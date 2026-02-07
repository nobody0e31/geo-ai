from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st

from geovision_ai.utils.config import DashboardConfig


def load_geojson_layers(geojson_dir: Path) -> dict:
    layers = {}
    for path in geojson_dir.glob("*.geojson"):
        layers[path.stem] = gpd.read_file(path)
    return layers


def render_map(layers: dict) -> None:
    st.subheader("Infrastructure Layers")
    if not layers:
        st.info("No GeoJSON layers found in the specified directory.")
        return

    layer_options = st.multiselect(
        "Select layers to display",
        options=list(layers.keys()),
        default=list(layers.keys())[:1],
    )

    if not layer_options:
        st.warning("Select at least one layer to display.")
        return

    selected = [layers[name] for name in layer_options]
    merged = gpd.GeoDataFrame(pd.concat(selected, ignore_index=True)) if len(selected) > 1 else selected[0]
    centroid = merged.unary_union.centroid
    view_state = pdk.ViewState(latitude=centroid.y, longitude=centroid.x, zoom=16)

    deck_layers = []
    for name in layer_options:
        gdf = layers[name]
        deck_layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=gdf.__geo_interface__,
                pickable=True,
                filled=True,
                get_fill_color=[0, 128, 255, 120],
                get_line_color=[0, 0, 0, 200],
                line_width_min_pixels=1,
            )
        )

    st.pydeck_chart(pdk.Deck(layers=deck_layers, initial_view_state=view_state, tooltip={"text": "{label}"}))


def render_metrics(metrics_path: Path) -> None:
    st.subheader("Model Metrics")
    if not metrics_path.exists():
        st.info("Metrics file not found.")
        return
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    st.json(metrics)


def parse_args() -> DashboardConfig:
    parser = argparse.ArgumentParser(description="GeoVision AI dashboard")
    parser.add_argument("--geojson", dest="geojson_dir", type=Path, required=True)
    parser.add_argument("--metrics", dest="metrics_path", type=Path, required=True)
    args = parser.parse_args()
    return DashboardConfig(**vars(args))


def main() -> None:
    config = parse_args()
    st.set_page_config(page_title="GeoVision AI", layout="wide")
    st.title("GeoVision AI | Rural Infrastructure Intelligence")
    st.caption("Map-first dashboard for monitoring AI-driven infrastructure extraction.")

    layers = load_geojson_layers(config.geojson_dir)
    render_map(layers)
    render_metrics(config.metrics_path)


if __name__ == "__main__":
    main()
