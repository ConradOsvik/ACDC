# ACDC — Automatic Corrosion Detection Cycle

An end-to-end pipeline for ML-assisted corrosion segmentation on infrastructure imagery. ACDC combines Label Studio, SAM 3, and YOLO26-seg into an iterative annotation and training loop: SAM 3 bootstraps the dataset using open-vocabulary text prompts, and a fine-tuned YOLO26-seg model progressively takes over as the annotation backend.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Training Workflow](#training-workflow)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Development](#development)

---

## Overview

The pipeline has three phases that repeat iteratively:

1. **Annotate** — An ML backend (SAM 3 or YOLO26-seg) generates a draft segmentation mask when an annotator opens a task in Label Studio. The annotator reviews and accepts or corrects it.
2. **Train** — Clicking "Start Training" in the Label Studio UI triggers the YOLO26-seg backend to export the current annotations, run a training pass, and hot-reload the new weights — all without leaving the browser.
3. **Evaluate** — `ls-cli` computes per-class IoU, Dice, Recall, and mAP@50 for YOLO26-seg and SAM 3 on a held-out test set.

The two label classes are **Red Rust** (iron oxide on steel) and **White Rust** (zinc oxide on galvanised steel).

---

## Architecture

```
┌──────────────────────────── ls-net ───────────────────────────────┐
│                                                                    │
│  ┌──────────────┐  SQL   ┌───────────────┐                        │
│  │   postgres   │◄──────►│ label-studio  │  :8080                 │
│  │ (16-alpine)  │        │ (heartex/ls)  │                        │
│  └──────────────┘        └──────┬────────┘                        │
│                        /predict │ START_TRAINING webhook           │
│                    ┌────────────┴──────┐                          │
│                    ▼                   ▼                          │
│             ┌────────────┐    ┌──────────────┐                   │
│             │  yolo-seg  │    │     sam3     │                   │
│             │   :9090    │    │    :9090     │                   │
│             └────────────┘    └──────────────┘                   │
└────────────────────────────────────────────────────────────────────┘
         ▲
         │  docker socket + Label Studio REST API
    ┌─────────┐
    │ ls-cli  │  (runs on host)
    └─────────┘
```

| Service | Image | Host port |
|---|---|---|
| `label-studio` | `heartexlabs/label-studio:latest` | 8080 |
| `postgres` | `postgres:16-alpine` | — (internal only) |
| `yolo-seg` | built from `ml_backends/yolo/Dockerfile` | 9090 |
| `sam3` | built from `ml_backends/sam3/Dockerfile` | 9091 |

All services communicate over the `ls-net` Docker network. The CLI talks to Label Studio via its REST API and to containers via the Docker socket.

---

## Prerequisites

- **Docker** ≥ 24 with the Compose plugin (`docker compose version`)
- **Python** ≥ 3.12
- **[uv](https://docs.astral.sh/uv/)** — `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **NVIDIA GPU + drivers** (optional, for GPU-accelerated training and SAM 3 inference)
  - NVIDIA Container Toolkit must be installed for GPU Compose variants

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd project
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — at minimum you need `LABEL_STUDIO_API_KEY` (see step 4).

### 3. Start the stack and bootstrap the project

```bash
# CPU
ls-cli init

# GPU (uses GPU variants for both model backends)
ls-cli init --gpu
```

`init` starts all services, waits for Label Studio to become healthy, then pauses and walks you through pasting the API token from the Label Studio UI. Once the token is validated it creates a project pre-configured with Red Rust and White Rust brush labels and attaches the YOLO26-seg backend.

### 4. Upload images and start annotating

Upload images through the Label Studio UI. The SAM 3 backend will generate draft masks on-the-fly when tasks are opened. Review, correct, and submit each annotation.

---

## Configuration

All configuration lives in `.env`. Copy `.env.example` to get started.

| Variable | Default | Description |
|---|---|---|
| `LABEL_STUDIO_API_KEY` | *(required after first boot)* | JWT token from Label Studio → Account & Settings |
| `LABEL_STUDIO_USERNAME` | `admin@example.com` | Admin account created on first boot |
| `LABEL_STUDIO_PASSWORD` | `changeme` | Admin password created on first boot |
| `YOLO_DEFAULT_MODEL` | `yolo26m-seg.pt` | Base weights to train from. Variants: `n/s/m/l/x` |
| `YOLO_CONF` | `0.25` | Confidence threshold for predictions |
| `YOLO_TRAIN_EPOCHS` | `50` | Training epochs per run |
| `YOLO_TRAIN_IMGSZ` | `640` | Input image size |
| `YOLO_TRAIN_BATCH` | `-1` | Batch size (`-1` = auto) |
| `YOLO_TRAIN_LR0` | `0.01` | Initial learning rate |
| `YOLO_TRAIN_FREEZE` | `0` | Number of backbone layers to freeze |
| `YOLO_URL` | `http://yolo-seg:9090` | Override YOLO backend URL (rarely needed) |
| `SAM3_URL` | `http://sam3:9090` | Override SAM 3 backend URL (rarely needed) |

Changing training hyperparameters takes effect on the next training run — no container restart needed. Changing `YOLO_DEFAULT_MODEL` requires restarting `yolo-seg`.

### SAM 3 model weights

The SAM 3 container expects the model weights at `ml_backends/sam3/sam3.pt`. This file is mounted read-only into the container at startup. Download or place the weights there before running `ls-cli up`.

---

## CLI Reference

Install once with `uv sync`, then use `ls-cli` from anywhere inside the project.

### Stack lifecycle

```bash
ls-cli init [--gpu]               # Full bootstrap: start stack + create project + attach backend
ls-cli up   [--gpu] [--build]     # Start Label Studio + ML backends
ls-cli down                        # Stop all services
```

### Project management

```bash
ls-cli project create [--title TEXT] [--backend yolo|sam3|none]
ls-cli project list
ls-cli project delete PROJECT_ID
```

### Backend management

```bash
ls-cli backend list [--project PROJECT_ID]
ls-cli backend switch PROJECT_ID yolo|sam3|none   # Switch active backend immediately
```

### YOLO backend

```bash
# Trigger training from the CLI (same as clicking "Start Training" in the UI)
ls-cli yolo train [--project-id ID] [--test-split 0.1] [--max-images 200]

# Deploy a specific weights file into the container
ls-cli yolo deploy path/to/weights.pt

# Show container status and current environment
ls-cli yolo status

# Export annotations as a YOLO dataset to disk
ls-cli yolo export [--project-id ID] [--output-dir DIR] [--split 0.8] [--test-split 0.1]

# Visualise a dataset export (image + polygon overlays)
ls-cli yolo visualize [--dataset-dir DIR] [--split all|train|val] [--save-dir DIR]

# Evaluate YOLO predictions against ground truth
ls-cli yolo evaluate [--project-id ID] [--max-images N] [--run RUN_NAME] [--export results.json]

# List all training runs with their key metrics
ls-cli yolo runs [--sort-by mtime|mAP50|mAP50-95|epochs]

# Show detailed metrics for a run and optionally copy artefacts
ls-cli yolo metrics [--run RUN_NAME] [--export DIR]
```

### SAM 3 backend

```bash
# Evaluate SAM 3 predictions against ground truth
ls-cli sam3 evaluate [--project-id ID] [--max-images N] [--run RUN_NAME] [--export results.json]
```

### Comparison and dataset analysis

```bash
# Side-by-side visual comparison of model predictions vs ground truth
ls-cli compare visualize [--run RUN [--run RUN ...]] [--sam3] [--task-id ID] [--max-images N]

# Dataset statistics (class distribution, mask sizes, co-occurrence)
ls-cli dataset stats [--project-id ID] [--export stats.json]
```

---

## Training Workflow

The intended annotation cycle:

```
Upload images
     │
     ▼
Annotate with SAM 3 (open-vocabulary text prompts: "red rust", "white rust")
     │
     ├─── Enough annotations? ──No──► Keep annotating
     │
     ▼
ls-cli yolo train --test-split 0.1
     │
     ▼
ls-cli backend switch <project-id> yolo
     │
     ▼
Continue annotating — YOLO26-seg generates drafts, SAM 3 on fallback
     │
     ▼
Repeat from training step as dataset grows
```

Each training run saves weights to `runs/yolo/train_YYYYMMDD_HHMMSS/weights/best.pt` on the host (mounted from `/data/runs` inside the container). The backend auto-discovers the newest weights on the next prediction request.

### Training is always from the base checkpoint

To prevent overfitting on small datasets and avoid managing a checkpoint chain, every training run starts from `YOLO_DEFAULT_MODEL` rather than the most recently saved weights. This is intentional.

---

## Evaluation

Evaluate both models on the same held-out images:

```bash
# Evaluate the current YOLO model
ls-cli yolo evaluate --test-split 0.1 --export exports/yolo_results.json

# Evaluate SAM 3 on the same test split from the latest training run
ls-cli sam3 evaluate --export exports/sam3_results.json

# Compare runs visually
ls-cli compare visualize --run train_20250510_143022 --sam3 --max-images 20
```

Metrics reported: per-class and mean **IoU**, **Dice coefficient**, **Precision**, **Recall**, and **mAP@50**.

---

## Project Structure

```
project/
├── .env.example                  # All configurable variables with documentation
├── pyproject.toml                # Python package — installs ls-cli entry point
├── infra/
│   ├── dev/
│   │   ├── docker-compose.ui.yml         # Label Studio + Postgres
│   │   ├── docker-compose.models.yml     # YOLO + SAM 3 (CPU)
│   │   └── docker-compose.models.gpu.yml # YOLO + SAM 3 (GPU)
│   └── prod/
│       ├── docker-compose.ui.yml
│       ├── docker-compose.models.yml
│       └── docker-compose.models.gpu.yml
├── ml_backends/
│   ├── utils.py                  # Shared: image download, RLE decode, YOLO export, mask conversion
│   ├── yolo/
│   │   ├── Dockerfile
│   │   ├── model.py              # LabelStudioMLBase: predict + fit (training handler)
│   │   └── requirements.txt
│   └── sam3/
│       ├── Dockerfile
│       ├── model.py              # LabelStudioMLBase: predict (text + bbox modes)
│       ├── requirements.txt
│       └── sam3.pt               # Model weights — not in repo, place here before first run
├── src/ls_cli/
│   ├── main.py                   # Typer app and command registration
│   ├── config.py                 # Settings loaded from .env
│   ├── commands/
│   │   ├── stack.py              # init / up / down
│   │   ├── project.py            # project create / list / delete
│   │   ├── backend.py            # backend list / switch
│   │   ├── yolo.py               # yolo train / evaluate / runs / metrics / export / deploy
│   │   ├── sam3.py               # sam3 evaluate
│   │   ├── compare.py            # compare visualize
│   │   └── dataset.py            # dataset stats
│   └── utils/
│       ├── docker.py             # compose_up/down/restart, docker_exec, docker_cp
│       ├── dataset.py            # YOLO dataset export + visualisation wrappers
│       ├── backends.py           # BackendChoice enum, ensure_single_backend
│       └── metrics.py            # list_runs, export_run, metrics parsing
├── runs/
│   └── yolo/                     # Training run directories (host mount from container /data/runs)
│       └── train_YYYYMMDD_HHMMSS/
│           ├── weights/best.pt
│           ├── results.csv
│           ├── test_metrics.json
│           └── ls_train_info.json
└── exports/                      # CLI export outputs (datasets, evaluation JSON, stats)
```

---

## Development

### Install with dev dependencies

```bash
uv sync
```

### Rebuild backend images after code changes

```bash
ls-cli up --build
```

Or rebuild a single service:

```bash
docker compose -f infra/dev/docker-compose.models.yml build yolo-seg
```

### Run backends without GPU

The CPU Compose files work on any machine with Docker. SAM 3 inference is slow on CPU (~10–30 s/image depending on hardware); the YOLO backend is fast.

### Switching between dev and prod

The `infra/dev/` and `infra/prod/` stacks are structurally identical; the production variants are intended for deployment on a server with persistent volumes and tighter credentials. The `ls-cli` commands default to `infra/dev/` — pass `--env-file` to point at a different `.env` if running both environments on the same host.

### Known limitations

- **Threading workaround** — The YOLO backend patches the Label Studio ML SDK's `SimpleJobManager` to use daemon threads instead of forked processes. PyTorch's CUDA context does not survive a `fork()`, causing silent training failures. This workaround works for single-user use but has not been tested under concurrent training requests.
- **Training always resets to base weights** — Each training run starts from `YOLO_DEFAULT_MODEL`, not from the previous run's weights. This avoids checkpoint chain management and prevents overfitting on small datasets, but discards incremental transfer benefit as the dataset grows.
- **RLE → polygon conversion is lossy** — Label Studio stores brush annotations as RLE bitmasks; YOLO trains on polygon contours. The conversion via OpenCV contour approximation introduces small boundary artefacts, visible as dashed lines between disconnected mask regions in the Label Studio UI.
