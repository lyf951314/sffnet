#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch SFFNet inference with JSON output.

Output format:
[
    {
        "image_id": 0,
        "category_id": 0,
        "bbox": [x, y, width, height],
        "score": 0.95
    }
]

Use --image-list to preserve the evaluation-set order. Otherwise, --image-dir
assigns image_id values in natural filename order (1.jpg, 2.jpg, 10.jpg).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from sffnet_inference import SFFNetDetector
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_bbox import decode_bbox, postprocess


IMAGE_EXTENSIONS = {
    ".bmp", ".dib", ".png", ".jpg", ".jpeg",
    ".pbm", ".pgm", ".ppm", ".tif", ".tiff",
}


def natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", str(path).replace("\\", "/").lower())
    return [int(x) if x.isdigit() else x for x in parts]


def collect_images(image_dir: str | None, image_list: str | None) -> list[Path]:
    if bool(image_dir) == bool(image_list):
        raise ValueError("Exactly one of --image-dir or --image-list must be provided.")

    if image_list:
        txt_path = Path(image_list).expanduser().resolve()
        if not txt_path.is_file():
            raise FileNotFoundError(f"Image list file not found: {txt_path}")

        paths = []
        for line_no, raw in enumerate(
            txt_path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Accepted image-list formats:
            # /path/a.jpg
            # /path/a.jpg x1,y1,x2,y2,class ...
            path = Path(line.split()[0]).expanduser()
            if not path.is_absolute():
                path = txt_path.parent / path
            path = path.resolve()

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image format on line {line_no}: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"Image not found on line {line_no}: {path}")

            paths.append(path)
        return paths

    root = Path(image_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Image directory not found: {root}")

    paths = [
        p.resolve()
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort(key=natural_sort_key)
    return paths


def predict_one_image(
    detector: SFFNetDetector,
    image: Image.Image,
) -> np.ndarray | None:
    """Return [top, left, bottom, right, score, class_id]."""
    image_shape = np.array(np.shape(image)[0:2], dtype=np.int64)
    image = cvtColor(image)

    resized = resize_image(
        image,
        (detector.input_shape[1], detector.input_shape[0]),
        detector.letterbox_image,
    )

    data = preprocess_input(np.array(resized, dtype=np.float32))
    data = np.expand_dims(np.transpose(data, (2, 0, 1)), 0)
    images = torch.from_numpy(np.asarray(data)).float()

    if detector.cuda:
        images = images.cuda(non_blocking=True)

    with torch.inference_mode():
        heatmap, size, offset = detector.net(images)

        decoded = decode_bbox(
            heatmap,
            size,
            offset,
            detector.confidence,
            detector.cuda,
        )

        results = postprocess(
            decoded,
            detector.nms,
            image_shape,
            detector.input_shape,
            detector.letterbox_image,
            detector.nms_iou,
        )

    if not results or results[0] is None or len(results[0]) == 0:
        return None
    return results[0]


def detections_to_json_records(
    detections: np.ndarray | None,
    image_id: int,
    image_width: int,
    image_height: int,
) -> list[dict]:
    """
    [top,left,bottom,right,score,class_id]
        ->
    image_id/category_id/bbox[x,y,w,h]/score
    """
    if detections is None or len(detections) == 0:
        return []

    # Match the reference JSON: sort detections within each image by descending score.
    detections = detections[
        np.argsort(-detections[:, 4], kind="stable")
    ]

    records = []
    for row in detections:
        top, left, bottom, right, score, class_id = [
            float(v) for v in row[:6]
        ]

        left = min(max(left, 0.0), float(image_width))
        right = min(max(right, 0.0), float(image_width))
        top = min(max(top, 0.0), float(image_height))
        bottom = min(max(bottom, 0.0), float(image_height))

        left_i, top_i = int(left), int(top)
        right_i, bottom_i = int(right), int(bottom)
        width_i = right_i - left_i
        height_i = bottom_i - top_i

        if width_i <= 0 or height_i <= 0:
            continue

        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(class_id),
                "bbox": [left_i, top_i, width_i, height_i],
                "score": float(score),
            }
        )

    return records



def save_visualization(
    image: Image.Image,
    detections: np.ndarray | None,
    output_path: Path,
    line_width: int = 3,
) -> Path:
    """
    Save detection visualizations. Draw bounding boxes only, without labels or scores.

    Each detections row is [top, left, bottom, right, score, class_id].
    """
    vis_image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis_image)

    if detections is not None and len(detections) > 0:
        image_width, image_height = vis_image.size
        for row in detections:
            top, left, bottom, right = [float(v) for v in row[:4]]

            left = min(max(left, 0.0), float(image_width - 1))
            right = min(max(right, 0.0), float(image_width - 1))
            top = min(max(top, 0.0), float(image_height - 1))
            bottom = min(max(bottom, 0.0), float(image_height - 1))

            if right <= left or bottom <= top:
                continue

            # Draw boxes only: no class names, class IDs, or confidence scores.
            draw.rectangle(
                [int(left), int(top), int(right), int(bottom)],
                outline=(0, 255, 0),
                width=max(1, int(line_width)),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the original image extension and save the visualization as RGB.
    vis_image.save(output_path)
    return output_path


def save_json(records: list[dict], output_path: str) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
        f.write("\n")

    return output


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run batch SFFNet inference and save predictions as JSON."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-dir", type=str, help="Directory containing test images.")
    source.add_argument(
        "--image-list",
        type=str,
        help="Text file listing test images; assign image_id in the listed order.",
    )

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--classes-path", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="full_ap_detections.json",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Enable additional class-wise NMS (disabled by default).",
    )
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument(
        "--letterbox",
        action="store_true",
        help="Enable letterbox resizing (direct resizing by default).",
    )
    parser.add_argument(
        "--vis-dir",
        type=str,
        default="predict_visualizations",
        help="Directory for detection visualizations (default: predict_visualizations).",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization output; write JSON only.",
    )
    parser.add_argument(
        "--box-width",
        type=int,
        default=3,
        help="Bounding-box line width in pixels (default: 3).",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be in [0, 1].")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in (0, 1].")
    if min(args.input_size) <= 0:
        raise ValueError("--input-size dimensions must be positive.")
    if any(x % 32 != 0 for x in args.input_size):
        raise ValueError("SFFNet input height and width must be multiples of 32.")
    if args.box_width < 1:
        raise ValueError("--box-width must be at least 1.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    image_paths = collect_images(args.image_dir, args.image_list)

    if not image_paths:
        output = save_json([], args.output)
        print(f"No images found; saved an empty JSON file: {output}")
        return 0

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    if not args.cpu and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.", file=sys.stderr)

    detector = SFFNetDetector(
        model_path=str(Path(args.model_path).expanduser()),
        classes_path=str(Path(args.classes_path).expanduser()),
        backbone="sffnet",
        input_shape=list(args.input_size),
        confidence=float(args.confidence),
        nms=bool(args.nms),
        nms_iou=float(args.nms_iou),
        letterbox_image=bool(args.letterbox),
        cuda=use_cuda,
    )

    all_records = []
    vis_dir = Path(args.vis_dir).expanduser().resolve()
    visualized_count = 0

    for image_id, image_path in enumerate(
        tqdm(image_paths, desc="SFFNet prediction")
    ):
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()

        detections = predict_one_image(detector, image)

        all_records.extend(
            detections_to_json_records(
                detections,
                image_id=image_id,
                image_width=image.width,
                image_height=image.height,
            )
        )

        if not args.no_visualize:
            # Prefix filenames with image_id to avoid overwriting images with duplicate names.
            vis_name = f"{image_id:06d}_{image_path.stem}{image_path.suffix.lower()}"
            save_visualization(
                image=image,
                detections=detections,
                output_path=vis_dir / vis_name,
                line_width=args.box_width,
            )
            visualized_count += 1

    all_records.sort(
        key=lambda item: (item["image_id"], -item["score"])
    )

    output = save_json(all_records, args.output)

    print(f"Number of images: {len(image_paths)}")
    print(f"Number of detections: {len(all_records)}")
    print(f"JSON saved to: {output}")
    if not args.no_visualize:
        print(f"Number of visualized images: {visualized_count}")
        print(f"Visualization directory: {vis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

