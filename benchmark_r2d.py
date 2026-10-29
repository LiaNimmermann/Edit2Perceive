#!/usr/bin/env python3
"""Run Edit2Perceive normal estimation on the RDR2 dataset."""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/media/rdr2/RDR2_dataset_processed_test")
DEFAULT_MODEL_ROOT = SCRIPT_DIR / "FLUX.1-Kontext-dev"
DEFAULT_CHECKPOINT = SCRIPT_DIR / "ckpts/normal.safetensors"
DEFAULT_LORA_CHECKPOINT = SCRIPT_DIR / "ckpts/normal_lora.safetensors"
IMAGE_PATTERN = re.compile(r"o_(\d+)_(\d+)\Z")
METRIC_NAMES = (
    "mean",
    "median",
    "rmse",
    "accuracy_5",
    "accuracy_7_5",
    "accuracy_11_25",
    "accuracy_22_5",
    "accuracy_30",
)
THRESHOLDS = (5.0, 7.5, 11.25, 22.5, 30.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Edit2Perceive and evaluate RDR2 normal maps."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root containing PNG/ and NormalMap/ (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for prediction maps and benchmark metrics.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="FLUX.1-Kontext-dev model directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Edit2Perceive normal checkpoint; defaults to normal.safetensors.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Load a normal LoRA checkpoint instead of the full normal model.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device (default: cuda:0).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help="Square model input/output resolution; must be divisible by 16 (default: 768).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=8,
        help="Diffusion inference steps (default: 8, matching inference.py).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Inference seed used for each image (default: 42).",
    )
    parser.add_argument(
        "--ground-truth-encoding",
        choices=("signed", "zero-to-one"),
        default="zero-to-one",
        help="EXR vector encoding: signed XYZ in [-1,1], or XYZ remapped to [0,1].",
    )
    parser.add_argument(
        "--ground-truth-x-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip the ground-truth X component to match Edit2Perceive's normal convention.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optionally process only the first N paired images (useful for a smoke test).",
    )
    parser.add_argument(
        "--visualize-images",
        type=int,
        default=0,
        help="Save side-by-side input/ground-truth/prediction PNGs for the first N images.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Do not save per-image prediction arrays.",
    )
    return parser.parse_args()


def find_samples(dataset_root):
    image_dir = dataset_root / "PNG"
    normal_dir = dataset_root / "NormalMap"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not normal_dir.is_dir():
        raise FileNotFoundError(f"Normal-map directory not found: {normal_dir}")

    images = sorted(image_dir.glob("o_*.png"))
    if not images:
        raise FileNotFoundError(f"No o_<id>_<daytime>.png images found in {image_dir}")

    samples = []
    missing = []
    for image_path in images:
        match = IMAGE_PATTERN.fullmatch(image_path.stem)
        if match is None:
            raise ValueError(f"Unexpected image filename: {image_path.name}")
        identifier, _daytime = match.groups()
        normal_path = normal_dir / f"n_{identifier}.exr"
        if normal_path.is_file():
            samples.append((image_path, normal_path))
        else:
            missing.append((image_path.name, normal_path.name))

    if missing:
        examples = ", ".join(f"{image} -> {normal}" for image, normal in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} image(s) have no matching normal map; examples: {examples}"
        )
    return samples


def load_ground_truth(path, cv2, np, encoding, resolution, flip_x):
    normal_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if normal_bgr is None:
        try:
            import OpenEXR
        except ImportError as error:
            raise RuntimeError(
                f"Could not read {path} with OpenCV. Install the OpenEXR Python "
                "package (included in requirements.txt) or use OpenCV built with "
                "OpenEXR support."
            ) from error
        channels = OpenEXR.File(str(path)).channels()
        if "RGB" in channels:
            normal_xyz = channels["RGB"].pixels[..., :3].astype(
                "float32", copy=False
            )
        else:
            missing_channels = {"R", "G", "B"} - channels.keys()
            if missing_channels:
                raise ValueError(
                    f"Expected RGB or separate R, G, and B channels in {path}; "
                    f"missing {sorted(missing_channels)}"
                )
            normal_xyz = np.stack(
                (channels["R"].pixels, channels["G"].pixels, channels["B"].pixels),
                axis=-1,
            ).astype("float32", copy=False)
    else:
        if normal_bgr.ndim != 3 or normal_bgr.shape[2] < 3:
            raise ValueError(
                f"Expected at least three EXR channels in {path}, got {normal_bgr.shape}"
            )
        # OpenCV returns BGR; Edit2Perceive evaluates normals in XYZ order.
        normal_xyz = normal_bgr[..., :3][..., ::-1].astype("float32", copy=True)
    if encoding == "zero-to-one":
        normal_xyz = normal_xyz * 2.0 - 1.0
    if flip_x:
        normal_xyz[..., 0] *= -1.0
    normal_xyz = cv2.resize(
        normal_xyz,
        (resolution, resolution),
        interpolation=cv2.INTER_LINEAR_EXACT,
    )
    return normal_xyz


def get_metrics(errors, np):
    return {
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "accuracy_5": float(100.0 * (errors < 5.0).mean()),
        "accuracy_7_5": float(100.0 * (errors < 7.5).mean()),
        "accuracy_11_25": float(100.0 * (errors < 11.25).mean()),
        "accuracy_22_5": float(100.0 * (errors < 22.5).mean()),
        "accuracy_30": float(100.0 * (errors < 30.0).mean()),
    }


def random_guess_metrics():
    """Expected angular metrics for an independent uniform random unit vector."""
    return {
        "mean": 90.0,
        "median": 90.0,
        "rmse": math.sqrt(math.pi**2 / 2.0 - 2.0) * 180.0 / math.pi,
        "accuracy_5": 100.0 * (1.0 - math.cos(math.radians(5.0))) / 2.0,
        "accuracy_7_5": 100.0 * (1.0 - math.cos(math.radians(7.5))) / 2.0,
        "accuracy_11_25": 100.0 * (1.0 - math.cos(math.radians(11.25))) / 2.0,
        "accuracy_22_5": 100.0 * (1.0 - math.cos(math.radians(22.5))) / 2.0,
        "accuracy_30": 100.0 * (1.0 - math.cos(math.radians(30.0))) / 2.0,
    }


def create_metric_accumulator(np, histogram_edges):
    return {
        "histogram": np.zeros(histogram_edges.size - 1, dtype=np.int64),
        "error_sum": 0.0,
        "squared_error_sum": 0.0,
        "valid_pixels": 0,
        "threshold_counts": np.zeros(len(THRESHOLDS), dtype=np.int64),
    }


def update_metric_accumulator(accumulator, errors, histogram_edges, np):
    accumulator["histogram"] += np.histogram(errors, bins=histogram_edges)[0]
    accumulator["error_sum"] += float(errors.sum(dtype=np.float64))
    accumulator["squared_error_sum"] += float(
        np.square(errors.astype(np.float64)).sum()
    )
    accumulator["valid_pixels"] += errors.size
    accumulator["threshold_counts"] += np.asarray(
        [(errors < threshold).sum() for threshold in THRESHOLDS], dtype=np.int64
    )


def finalize_metric_accumulator(accumulator, histogram_edges, np):
    valid_pixels = accumulator["valid_pixels"]
    median_bin = int(
        np.searchsorted(
            np.cumsum(accumulator["histogram"]), (valid_pixels + 1) // 2
        )
    )
    metrics = {
        "mean": accumulator["error_sum"] / valid_pixels,
        "median": float(
            (histogram_edges[median_bin] + histogram_edges[median_bin + 1]) / 2
        ),
        "rmse": float((accumulator["squared_error_sum"] / valid_pixels) ** 0.5),
    }
    metrics.update(
        {
            name: float(100.0 * count / valid_pixels)
            for name, count in zip(
                METRIC_NAMES[3:], accumulator["threshold_counts"]
            )
        }
    )
    return metrics


def load_model(model_root, checkpoint_path, use_lora, device, torch_dtype):
    sys.path.insert(0, str(SCRIPT_DIR))
    from models.utils import load_state_dict, parse_flux_model_configs
    from pipelines.flux_image_new import FluxImagePipeline

    load_device = "cpu" if device.type == "cuda" else device
    working_directory = Path.cwd()
    os.chdir(SCRIPT_DIR)
    try:
        pipe = FluxImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=load_device,
            model_configs=parse_flux_model_configs(str(model_root)),
        )
    finally:
        os.chdir(working_directory)
    if use_lora:
        pipe.load_lora(pipe.dit, str(checkpoint_path), hotload=False)
    else:
        state_dict = load_state_dict(
            str(checkpoint_path), torch_dtype=torch_dtype, device=load_device
        )
        pipe.dit.load_state_dict(state_dict, assign=True)
        del state_dict
    if device.type == "cuda":
        pipe.device = device
        pipe.enable_vram_management()
    pipe.eval()
    return pipe


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else (
        DEFAULT_LORA_CHECKPOINT if args.lora else DEFAULT_CHECKPOINT
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = (SCRIPT_DIR / checkpoint_path).resolve()

    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be greater than zero")
    if args.visualize_images < 0:
        raise ValueError("--visualize-images cannot be negative")
    if args.resolution < 16 or args.resolution % 16:
        raise ValueError("--resolution must be a positive multiple of 16")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be greater than zero")
    if not model_root.is_dir():
        raise FileNotFoundError(f"FLUX.1-Kontext-dev model directory not found: {model_root}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Edit2Perceive checkpoint not found: {checkpoint_path}")

    samples = find_samples(dataset_root)
    if args.max_images is not None:
        samples = samples[: args.max_images]
    visualize_count = min(args.visualize_images, len(samples))

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from tqdm import tqdm
    from models.unified_dataset import UnifiedDataset

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Install CUDA-enabled PyTorch "
                "and verify that an NVIDIA GPU is visible."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device {device} requested, but only "
                f"{torch.cuda.device_count()} device(s) are available"
            )
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    transform = UnifiedDataset.default_image_operator(
        height=args.resolution, width=args.resolution
    )
    pipe = load_model(
        model_root, checkpoint_path, args.lora, device, torch_dtype
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    if not args.metrics_only:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    if visualize_count:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    histogram_edges = np.linspace(0.0, 180.0, 18001, dtype=np.float64)
    global_accumulator = create_metric_accumulator(np, histogram_edges)
    daytime_accumulators = {}
    csv_path = output_dir / "per_image_metrics.csv"
    fieldnames = (
        "image",
        "ground_truth",
        "prediction",
        "valid_pixels",
        *METRIC_NAMES,
    )

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        with torch.inference_mode():
            for index, (image_path, normal_path) in enumerate(
                tqdm(samples, desc="Benchmarking", unit="image"), start=1
            ):
                image_match = IMAGE_PATTERN.fullmatch(image_path.stem)
                if image_match is None:
                    raise ValueError(f"Unexpected image filename: {image_path.name}")
                identifier, daytime = image_match.groups()

                with Image.open(image_path) as image_file:
                    image_pil = image_file.convert("RGB")
                image_rgb = np.asarray(image_pil, dtype=np.uint8)
                model_input = transform(image_pil)
                prediction = pipe(
                    prompt="Transform to normal map while maintaining original composition",
                    kontext_images=model_input,
                    height=args.resolution,
                    width=args.resolution,
                    embedded_guidance=1,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                    output_type="np",
                    rand_device=str(device),
                    task="normal",
                )
                prediction = np.asarray(prediction, dtype=np.float32)
                if prediction.shape != (args.resolution, args.resolution, 3):
                    raise ValueError(
                        f"Unexpected prediction shape for {image_path.name}: "
                        f"{prediction.shape}"
                    )

                ground_truth = load_ground_truth(
                    normal_path,
                    cv2,
                    np,
                    args.ground_truth_encoding,
                    args.resolution,
                    args.ground_truth_x_flip,
                )
                pred_length = np.linalg.norm(prediction, axis=2)
                gt_length = np.linalg.norm(ground_truth, axis=2)
                valid = (
                    np.isfinite(prediction).all(axis=2)
                    & np.isfinite(ground_truth).all(axis=2)
                    & (pred_length > 1e-8)
                    & (gt_length > 1e-8)
                )
                if not np.any(valid):
                    raise ValueError(f"No valid ground-truth normals in {normal_path}")

                pred_unit = prediction[valid] / pred_length[valid, None]
                gt_unit = ground_truth[valid] / gt_length[valid, None]
                cosine = np.clip(np.sum(pred_unit * gt_unit, axis=1), -1.0, 1.0)
                errors = np.degrees(np.arccos(cosine))
                image_metrics = get_metrics(errors, np)

                if index <= visualize_count:
                    gt_visualization = np.zeros_like(ground_truth, dtype=np.float32)
                    prediction_visualization = np.zeros_like(
                        prediction, dtype=np.float32
                    )
                    gt_visualization[valid] = gt_unit
                    prediction_visualization[valid] = pred_unit
                    width = args.resolution
                    height = args.resolution
                    header_height = 36
                    visualization = Image.new(
                        "RGB", (width * 3, height + header_height), "white"
                    )
                    draw = ImageDraw.Draw(visualization)
                    draw.text((8, 10), "Input RGB", fill="black")
                    draw.text((width + 8, 10), "Ground truth normal", fill="black")
                    draw.text((width * 2 + 8, 10), "Edit2Perceive prediction", fill="black")
                    visualization.paste(image_pil.resize((width, height)), (0, header_height))
                    for panel_index, normal in enumerate(
                        (gt_visualization, prediction_visualization), start=1
                    ):
                        normal_rgb = np.clip((normal + 1.0) * 127.5, 0, 255).astype(
                            np.uint8
                        )
                        normal_rgb[~valid] = 0
                        visualization.paste(
                            Image.fromarray(normal_rgb, mode="RGB"),
                            (panel_index * width, header_height),
                        )
                    visualization.save(visualization_dir / f"{image_path.stem}.png")

                update_metric_accumulator(
                    global_accumulator, errors, histogram_edges, np
                )
                if daytime not in daytime_accumulators:
                    daytime_accumulators[daytime] = {
                        "metrics": create_metric_accumulator(np, histogram_edges),
                        "images": 0,
                        "identifiers": set(),
                    }
                daytime_accumulator = daytime_accumulators[daytime]
                update_metric_accumulator(
                    daytime_accumulator["metrics"], errors, histogram_edges, np
                )
                daytime_accumulator["images"] += 1
                daytime_accumulator["identifiers"].add(identifier)

                prediction_path = ""
                if not args.metrics_only:
                    prediction_path = str(prediction_dir / f"{image_path.stem}.npy")
                    np.save(prediction_path, prediction.astype(np.float32, copy=False))

                writer.writerow(
                    {
                        "image": image_path.name,
                        "ground_truth": normal_path.name,
                        "prediction": prediction_path,
                        "valid_pixels": int(errors.size),
                        **image_metrics,
                    }
                )
                csv_file.flush()

    global_metrics = finalize_metric_accumulator(
        global_accumulator, histogram_edges, np
    )
    daytime_metrics = {}
    daytime_csv_path = output_dir / "daytime_metrics.csv"
    with daytime_csv_path.open("w", newline="", encoding="utf-8") as daytime_csv_file:
        daytime_fieldnames = (
            "daytime",
            "images",
            "unique_ids",
            "valid_pixels",
            *METRIC_NAMES,
        )
        writer = csv.DictWriter(daytime_csv_file, fieldnames=daytime_fieldnames)
        writer.writeheader()
        for daytime in sorted(daytime_accumulators, key=int):
            accumulator = daytime_accumulators[daytime]
            metrics = finalize_metric_accumulator(
                accumulator["metrics"], histogram_edges, np
            )
            valid_pixels = accumulator["metrics"]["valid_pixels"]
            daytime_metrics[daytime] = {
                "images": accumulator["images"],
                "unique_ids": len(accumulator["identifiers"]),
                "valid_pixels": valid_pixels,
                "metrics": metrics,
            }
            writer.writerow(
                {
                    "daytime": daytime,
                    "images": accumulator["images"],
                    "unique_ids": len(accumulator["identifiers"]),
                    "valid_pixels": valid_pixels,
                    **metrics,
                }
            )
        baseline_metrics = random_guess_metrics()
        writer.writerow(
            {
                "daytime": "random_guess_uniform_sphere",
                "images": 0,
                "unique_ids": 0,
                "valid_pixels": 0,
                **baseline_metrics,
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "model": "Edit2Perceive",
        "model_root": str(model_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_type": "lora" if args.lora else "full",
        "device": str(device),
        "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "ground_truth_encoding": args.ground_truth_encoding,
        "ground_truth_x_flip": args.ground_truth_x_flip,
        "images_processed": len(samples),
        "valid_pixels": global_accumulator["valid_pixels"],
        "global_median_method": "0.01-degree histogram (within 0.01 degrees)",
        "metrics": global_metrics,
        "random_guess_baseline": {
            "assumption": "Independent uniform random unit vector on the sphere",
            "metrics": baseline_metrics,
        },
        "daytime_metrics": daytime_metrics,
        "visualized_images": visualize_count,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, allow_nan=False)
        summary_file.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"Per-image metrics: {csv_path}")
    print(f"Per-daytime metrics: {daytime_csv_path}")
    if not args.metrics_only:
        print(f"Predictions: {prediction_dir}")
    if visualize_count:
        print(f"Visualizations ({visualize_count}): {visualization_dir}")


if __name__ == "__main__":
    main()
