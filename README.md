# GeoVision AI

GeoVision AI is a production-ready, end-to-end pipeline for automated rural infrastructure extraction from drone imagery. It delivers GIS-ready vector layers for buildings, roads, water bodies, and micro-assets with a dual-stream deep learning architecture designed for national-scale deployments.

## Highlights
- Dual-stream pipeline: segmentation (buildings, roads, water) + classification (roof materials, micro-assets).
- Multi-task training with attention-ready blocks and precision-optimized loss.
- GPU-accelerated batch inference that outputs GeoJSON layers.
- Streamlit GIS-style dashboard for interactive review and progress monitoring.

## Repository Structure
```
src/geovision_ai/
  pipelines/            # preprocessing, training, validation, inference
  models.py              # segmentation + classification architectures
  preprocessing.py       # ortho-rectification hooks, tiling, augmentation
  geojson.py             # vectorization and GeoJSON export
  metrics.py             # precision & IoU scoring
  ui/dashboard.py        # GIS dashboard
  utils/
    io.py                # data loading helpers
    config.py            # runtime configuration
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Preprocessing
```bash
geovision-preprocess \
  --input data/raw \
  --output data/processed \
  --tile-size 1024 \
  --stride 768
```

### Expected Processed Data Layout
```
data/processed/
  tiles/<scene_name>/tile_000001.npy
  masks/<scene_name>/tile_000001.npy
  labels.csv   # columns: tile,label (label is integer class id)
```

## Training
```bash
geovision-train \
  --data data/processed \
  --output artifacts/models \
  --epochs 50 \
  --batch-size 4 \
  --device cuda
```

## Validation
```bash
geovision-validate \
  --data data/processed \
  --checkpoint artifacts/models/segmentation.pt \
  --device cuda
```

## Inference + GeoJSON Export
```bash
geovision-infer \
  --input data/raw \
  --checkpoint-seg artifacts/models/segmentation.pt \
  --checkpoint-cls artifacts/models/classification.pt \
  --output artifacts/geojson \
  --device cuda
```

## Dashboard
```bash
geovision-dashboard \
  --geojson artifacts/geojson \
  --metrics artifacts/metrics.json
```

## Deployment Notes
- GeoTIFF imagery with correct CRS is strongly recommended for accurate vector outputs.
- For S3 ingestion, mount buckets via AWS CLI or configure your data pipelines to place imagery into `data/raw`.

## License
Apache-2.0
