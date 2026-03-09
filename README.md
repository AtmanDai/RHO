# RHO: Robust Visual Localization on Maps with Panoramic Images

<p align="center">
  <img src="assets/teaser.svg" width="80%"/>
</p>

RHO is a visual localization system that estimates the precise position and orientation of a camera on a map using panoramic street-level images and OpenStreetMap data. It extends [OrienterNet](https://github.com/facebookresearch/OrienterNet) with panoramic image support, fusing three consecutive 120° field-of-view images into a unified 360° bird's-eye-view (BEV) representation for improved localization accuracy.

The system works by encoding query images and rasterized OpenStreetMap tiles into a shared feature space, then performing template matching to jointly estimate location and heading. RHO supports both single-image and sequential (trajectory-based) localization, and includes extensive robustness evaluation under challenging conditions such as night, rain, fog, snow, and motion blur.

## Key Features

- **Panoramic localization**: Fuses 3 × 120° FOV images into a 360° BEV representation
- **Map-based matching**: Leverages OpenStreetMap semantic data (areas, ways, nodes) as spatial priors
- **Sequential mode**: Temporal trajectory alignment with rigid alignment for improved accuracy
- **Robustness evaluation**: Includes evaluation under adverse weather/lighting (night, rain, fog, snow, motion blur, over/under exposure)
- **Multi-dataset support**: Mapillary (panoramic and standard), KITTI, and Sim2Real datasets
- **Image synthesis**: Scripts to generate augmented images for robustness testing (night, rain, fog, snow, sensor noise)

## Architecture

RHO consists of the following components:

| Component         | Description                                                  |
|-------------------|--------------------------------------------------------------|
| **Image Encoder** | ResNet-101 with FPN for extracting multi-scale image features |
| **BEV Projection**| Projects image features into bird's-eye-view space            |
| **Map Encoder**   | VGG-19 based encoder for rasterized OpenStreetMap tiles       |
| **BEV Net**       | 4-block network processing BEV features with confidence output|
| **Template Matching** | FFT-based cross-correlation with 64 rotations and 33 scale bins |

## Installation

### Requirements

- Python >= 3.8
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/AtmanDai/RHO.git
cd RHO

# Install the package and dependencies
pip install -e .
pip install -r requirements/full.txt
```

For running the demo with image calibration and geocoding support:

```bash
pip install -r requirements/demo.txt
```

For development (linting and formatting):

```bash
pip install -r requirements/dev.txt
```

## Data Preparation

RHO uses the following datasets:

- **Mapillary Geo-Localization (MGL)**: Street-level panoramic images across 7 cities (Berlin, Chicago, Detroit, Montrouge, San Francisco, Toulouse, Washington)
- **KITTI**: Autonomous driving sequences with 120° FOV perspective images
- **Sim2Real**: Synthetic-to-real domain adaptation dataset

The CV-RHO dataset extends the Mapillary MGL data with panoramic image sequences and robustness variants. Data and pretrained models can be downloaded from:
```
https://cvg-data.inf.ethz.ch/OrienterNet_CVPR2023
```

## Evaluation

### Mapillary Panoramic Evaluation

Evaluate the RHO model on the Mapillary panoramic dataset:

```bash
# Single-image evaluation
python -m maploc.evaluation.mapillary_pano \
    --experiment <path_to_checkpoint> \
    --split val

# Sequential evaluation (trajectory alignment)
python -m maploc.evaluation.mapillary_pano \
    --experiment <path_to_checkpoint> \
    --split val \
    --sequential
```

Optional arguments:
- `--output_dir <dir>`: Save visualization outputs
- `--num <n>`: Limit evaluation to `n` samples

### KITTI Evaluation

```bash
# Single-image evaluation
python -m maploc.evaluation.kitti \
    --experiment <path_to_checkpoint> \
    --split test

# Sequential evaluation
python -m maploc.evaluation.kitti \
    --experiment <path_to_checkpoint> \
    --split test \
    --sequential
```

### Robustness Evaluation

RHO includes evaluation scripts for various adverse conditions. Replace `<condition>` with one of: `night`, `rainy`, `foggy`, `snowy`, `motion_blur`, `over_exposure`, `under_exposure`.

```bash
# Panoramic robustness evaluation
python -m maploc.evaluation.mapillary_pano_<condition> \
    --experiment <path_to_checkpoint> \
    --split val

# Standard (120° FOV) robustness evaluation
python -m maploc.evaluation.mapillary_<condition> \
    --experiment <path_to_checkpoint> \
    --split val
```

### Evaluation Metrics

The evaluation reports recall at (1, 3, 5) meter/degree thresholds for:

| Metric | Description |
|--------|-------------|
| `xy_max_error` | Localization error (meters) |
| `yaw_max_error` | Heading error (degrees) |
| `xy_gps_error` | GPS-fused localization error |
| `xy_seq_error` | Sequential localization error (with trajectory alignment) |
| `yaw_seq_error` | Sequential heading error |
| `directional_error` | Directional localization error (KITTI) |

## Fine-Tuning and Training

### Training RHO (Panoramic)

Train the RHO model on panoramic images using [Hydra](https://hydra.cc/) for configuration:

```bash
python -m maploc.train_pano experiment.name=<experiment_name>
```

### Training OrienterNet (120° FOV)

Train the OrienterNet variant on standard perspective images:

```bash
python -m maploc.train_120 experiment.name=<experiment_name>
```

### Configuration Overrides

All training parameters can be overridden via Hydra's command-line syntax:

```bash
python -m maploc.train_pano \
    experiment.name=my_experiment \
    experiment.gpus=2 \
    training.lr=5e-6 \
    training.trainer.max_epochs=50 \
    data.loading.train.batch_size=8
```

### Default Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-5 |
| Weight decay | 1e-5 |
| LR scheduler | ReduceLROnPlateau (factor=0.7, patience=2) |
| Max epochs | 30 |
| Batch size | 36 |
| GPUs | 4 |
| Validation interval | Every 500 steps |
| Checkpointing | Top 3 by validation loss |

### Fine-Tuning from a Checkpoint

To fine-tune from an existing checkpoint:

```bash
python -m maploc.train_pano \
    experiment.name=finetune_experiment \
    training.finetune_from_checkpoint=<path_to_checkpoint>
```

Training automatically resumes from the last checkpoint if one is found in the experiment directory.

### Multi-GPU and Distributed Training

The training scripts support distributed data parallel (DDP) training. For SLURM-based multi-node training, the scripts automatically detect `SLURM_NNODES` and configure the trainer accordingly.

## Demo

Run the interactive demo for localizing query images:

```python
from maploc.demo import Demo

demo = Demo(experiment_or_path="OrienterNet_MGL")

# Localize an image using an address prior
image, camera, gravity, proj, bbox = demo.read_input_image(
    "path/to/image.jpg",
    prior_address="Times Square, New York"
)
data = demo.prepare_data(image, camera, gravity, bbox.canvas())
xy, yaw, prob, features, img = demo.localize(**data)
```

## Visualization

Jupyter notebooks are provided for visualizing predictions:

- `notebooks/visualize_predictions_mgl.ipynb` — Mapillary predictions
- `notebooks/visualize_predictions_kitti.ipynb` — KITTI predictions
- `notebooks/visualize_predictions_sequences.ipynb` — Sequential predictions

Visualization scripts for generating prediction plots:

```bash
# Panoramic prediction visualization
python -m visualization.viz_pred_pano

# 120° FOV prediction visualization
python -m visualization.viz_pred_120fov
```

## Project Structure

```
RHO/
├── maploc/                    # Main package
│   ├── models/                # Model architectures (RHO, OrienterNet)
│   ├── data/                  # Dataset loading (Mapillary, KITTI)
│   ├── evaluation/            # Evaluation scripts and metrics
│   ├── osm/                   # OpenStreetMap integration
│   ├── conf/                  # Hydra configuration files
│   ├── utils/                 # Utilities (geometry, visualization, I/O)
│   ├── train_pano.py          # Panoramic model training
│   ├── train_120.py           # 120° FOV model training
│   ├── demo.py                # Interactive demo
│   └── module.py              # PyTorch Lightning module
├── image_generation/          # Image augmentation (night, rain, fog, snow)
├── visualization/             # Visualization scripts
├── notebooks/                 # Jupyter notebooks
├── requirements/              # Dependency specifications
└── assets/                    # Static assets
```

## License

- The **CV-RHO dataset** is made available under the [CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) license following the data available on the Mapillary platform.
- The **model implementation and pre-trained weights** follow a [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license.
- **OpenStreetMap data** is licensed under the [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).

## Acknowledgments

This project is built upon [OrienterNet](https://github.com/facebookresearch/OrienterNet) and adapted from [PixLoc](https://github.com/cvg/pixloc). Developed by Paul-Edouard Sarlin and Ruize Dai at Meta Platforms, Inc.
