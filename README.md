# Edit2Perceive RDR2 Normal Benchmark

This directory adds an RDR2 surface-normal benchmark for Edit2Perceive. It runs
the pretrained Edit2Perceive normal estimator on images in `PNG/`, compares the
predictions with the matching OpenEXR maps in `NormalMap/`, and reports angular
error overall and by daytime.

## Requirements

- Python 3.12 (recommended by Edit2Perceive).
- An NVIDIA GPU and a CUDA-compatible PyTorch installation for practical
  inference. The benchmark enables Edit2Perceive's CPU offloading to reduce
  peak GPU memory use.
- The Edit2Perceive dependencies, FLUX.1-Kontext-dev base model, and a
  pretrained Edit2Perceive normal checkpoint.
- The RDR2 benchmark dataset.

Run commands from this directory:

```bash
cd E2P/Edit2Perceive
```

Install dependencies in a Python environment using the upstream requirements:

```bash
pip install -r requirements.txt
```

## Model files

Download the FLUX.1-Kontext-dev files described in
[README_E2P.md](README_E2P.md) into `FLUX.1-Kontext-dev/`. The benchmark uses
the full normal checkpoint by default:

```text
FLUX.1-Kontext-dev/
├── ae.safetensors
├── flux1-kontext-dev.safetensors
├── text_encoder/
│   └── model.safetensors
└── text_encoder_2/

ckpts/
└── normal.safetensors
```

The base model directory and full checkpoint are already the script defaults.
To benchmark the smaller normal LoRA checkpoint instead, download
`ckpts/normal_lora.safetensors` and add `--lora`.

## Dataset layout

The dataset root must contain both `PNG/` and `NormalMap/` directories:

```text
<dataset-root>/
├── PNG/
│   └── o_<id>_<daytime>.png
└── NormalMap/
    └── n_<id>.exr
```

Each input PNG must have a matching EXR for its ID. The benchmark evaluates
each daytime image against that ID's normal map. The dataset is not included in
this repository.

## Run the benchmark

Run the full evaluation and choose an output directory:

```bash
python benchmark_r2d.py \
  --dataset-root /path/to/RDR2_dataset_processed_test \
  --output-dir /path/to/rdr2-edit2perceive-results
```

Run a short smoke test before processing the full dataset:

```bash
python benchmark_r2d.py \
  --dataset-root /path/to/RDR2_dataset_processed_test \
  --output-dir /path/to/rdr2-edit2perceive-smoke-test \
  --max-images 5 \
  --metrics-only
```

To use the normal LoRA weights:

```bash
python benchmark_r2d.py \
  --dataset-root /path/to/RDR2_dataset_processed_test \
  --output-dir /path/to/rdr2-edit2perceive-lora-results \
  --lora
```

Main options:

| Option | Description |
| --- | --- |
| `--dataset-root PATH` | Dataset directory containing `PNG/` and `NormalMap/`. |
| `--output-dir PATH` | Required directory for metrics, predictions, and visualizations. |
| `--model-root PATH` | FLUX.1-Kontext-dev directory (defaults to the included model directory). |
| `--checkpoint PATH` | Override the full or LoRA normal checkpoint path. |
| `--lora` | Load LoRA weights; defaults to `ckpts/normal_lora.safetensors`. |
| `--device DEVICE` | Inference device, default `cuda:0`. |
| `--resolution N` | Square input/output resolution, default `768`; must be divisible by 16. |
| `--num-inference-steps N` | Number of diffusion steps, default `8`. |
| `--seed N` | Fixed inference seed used for each image, default `42`. |
| `--ground-truth-encoding {signed,zero-to-one}` | EXR vector encoding; default `zero-to-one`. |
| `--no-ground-truth-x-flip` | Disable the default X-axis flip applied to match Edit2Perceive's normal convention. |
| `--max-images N` | Process only the first N paired images. |
| `--visualize-images N` | Save side-by-side panels for the first N images. |
| `--metrics-only` | Do not save per-image prediction arrays. |

Edit2Perceive's inference preprocessing resizes each input image to a square,
defaulting to 768 by 768 pixels. Ground-truth normals are resized to the same
resolution before evaluation. EXR vector channels are reordered from OpenCV's
BGR order to XYZ. By default, values are decoded from `[0, 1]` to `[-1, 1]`
and the ground-truth X component is flipped, matching Edit2Perceive's normal
evaluation convention. Use `--ground-truth-encoding signed` for EXRs already
stored in `[-1, 1]`, or `--no-ground-truth-x-flip` if your maps use another
coordinate convention.

## Results

The output directory contains:

- `per_image_metrics.csv` — angular metrics for each image.
- `daytime_metrics.csv` — metrics grouped by daytime, plus an analytic
  random-guess baseline.
- `summary.json` — run configuration and overall/daytime metrics.
- `predictions/` — float32 XYZ prediction arrays with shape
  `(resolution, resolution, 3)`, unless `--metrics-only` was used.
- `visualizations/` — input, ground-truth, and prediction panels when
  `--visualize-images N` is greater than zero.

The global and daytime medians are computed from a 0.01-degree histogram. Use
`--metrics-only` to avoid writing prediction arrays when disk space is limited.

For the upstream project, training instructions, and other evaluation tasks,
see [README_E2P.md](README_E2P.md).