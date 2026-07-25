#!/usr/bin/env python3
"""Extract CTransPath features at existing MRePath/CLAM patch coordinates."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "tcga_coadread" / "raw_svs"
DEFAULT_COORD_DIR = (
    ROOT / "data" / "tcga_coadread" / "clam_20x_resnet50" / "patches"
)
DEFAULT_OUTPUT = (
    Path("/home/administrator/.cache/mrepath/tcga_coadread")
    / "clam_20x_ctranspath_samecoords"
)
DEFAULT_CHECKPOINT = (
    Path("/home/administrator/.cache/mrepath/models/ctranspath")
    / "ctranspath.pth"
)
DEFAULT_TRANSPATH_REPO = (
    Path("/home/administrator/.cache/mrepath/third_party/TransPath")
)
DEFAULT_CLAM_REPO = ROOT / "third_party" / "CLAM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--coord-dir", type=Path, default=DEFAULT_COORD_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--transpath-repo", type=Path, default=DEFAULT_TRANSPATH_REPO)
    parser.add_argument("--clam-repo", type=Path, default=DEFAULT_CLAM_REPO)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slide-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--status-file", default="ctranspath_status.csv")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def discover_slides(source: Path) -> dict[str, Path]:
    extensions = {".svs", ".tif", ".tiff", ".ndpi"}
    slides: dict[str, Path] = {}
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        previous = slides.setdefault(path.stem.lower(), path)
        if previous != path:
            raise RuntimeError(
                f"Duplicate slide stem {path.stem!r}: {previous} and {path}"
            )
    return slides


def validate_feature_file(path: Path, expected_rows: int) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    try:
        with h5py.File(path, "r") as handle:
            features = handle["features"]
            coords = handle["coords"]
            if (
                features.shape != (expected_rows, 768)
                or coords.shape != (expected_rows, 2)
                or features.dtype != np.float32
            ):
                return 0, 0
            return int(features.shape[0]), int(features.shape[1])
    except (OSError, KeyError):
        return 0, 0


def collate_features(
    batch: list[tuple[torch.Tensor, np.ndarray]],
) -> tuple[torch.Tensor, np.ndarray]:
    images = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return images, coords


def save_feature_batch(
    path: Path, features: np.ndarray, coords: np.ndarray, mode: str
) -> None:
    with h5py.File(path, mode) as handle:
        for key, values in (("features", features), ("coords", coords)):
            if key not in handle:
                maxshape = (None,) + values.shape[1:]
                handle.create_dataset(
                    key,
                    data=values,
                    maxshape=maxshape,
                    chunks=True,
                )
            else:
                dataset = handle[key]
                old_size = dataset.shape[0]
                dataset.resize(old_size + values.shape[0], axis=0)
                dataset[old_size:] = values


def build_model(
    device: torch.device, transpath_repo: Path, checkpoint: Path
) -> torch.nn.Module:
    sys.path.insert(0, str(transpath_repo))
    from ctran import ctranspath

    model = ctranspath()
    model.head = torch.nn.Identity()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device)
    model.eval()
    return model


def extract_slide(
    *,
    slide_path: Path,
    coord_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    workers: int,
    clam_repo: Path,
) -> tuple[int, int]:
    # CLAM's ``datasets`` directory is a namespace package, while MRePath has
    # a regular package with the same name.  Load this standalone module by
    # path so Python cannot resolve it to MRePath's package.
    dataset_module_path = clam_repo / "datasets" / "dataset_h5.py"
    spec = importlib.util.spec_from_file_location(
        "clam_dataset_h5", dataset_module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CLAM dataset module: {dataset_module_path}")
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    Whole_Slide_Bag_FP = dataset_module.Whole_Slide_Bag_FP

    temporary = output_path.with_suffix(f".h5.partial.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(str(slide_path))
    dataset = Whole_Slide_Bag_FP(
        file_path=str(coord_path),
        wsi=slide,
        pretrained=True,
        target_patch_size=224,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_features,
        persistent_workers=workers > 0,
    )

    mode = "w"
    try:
        with torch.inference_mode():
            for batch_index, (images, coords) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                features = model(images).float().cpu().numpy()
                save_feature_batch(temporary, features, coords, mode)
                mode = "a"
                if batch_index % 25 == 0 or batch_index == len(loader):
                    print(
                        f"[features] {slide_path.stem}: "
                        f"batch {batch_index}/{len(loader)}",
                        flush=True,
                    )
    finally:
        slide.close()

    os.replace(temporary, output_path)
    rows, dimension = validate_feature_file(output_path, len(dataset))
    if (rows, dimension) != (len(dataset), 768):
        raise RuntimeError(
            f"Feature validation failed: got {rows}x{dimension}, "
            f"expected {len(dataset)}x768"
        )
    return rows, dimension


def main() -> int:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    if Path(args.status_file).name != args.status_file:
        raise ValueError("--status-file must be a basename")

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")

    coord_paths = sorted(args.coord_dir.resolve().glob("*.h5"))
    if args.slide_id:
        coord_paths = [
            path
            for path in coord_paths
            if args.slide_id.lower() in path.stem.lower()
        ]
    if args.limit is not None:
        coord_paths = coord_paths[: args.limit]
    coord_paths = coord_paths[args.shard_index :: args.num_shards]
    if not coord_paths:
        raise RuntimeError("No coordinate HDF5 files selected")

    slides = discover_slides(args.source.resolve())
    missing_slides = [
        path.stem for path in coord_paths if path.stem.lower() not in slides
    ]
    if missing_slides:
        raise RuntimeError(
            f"Missing raw WSI for {len(missing_slides)} coordinate files: "
            f"{missing_slides[:5]}"
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / args.status_file
    rows: list[dict[str, Any]] = []
    if status_path.is_file():
        rows = pd.read_csv(status_path).fillna("").to_dict("records")
    by_slide = {str(row["slide_id"]): row for row in rows}

    model = build_model(
        device,
        args.transpath_repo.resolve(),
        args.checkpoint.resolve(),
    )
    print(
        f"[setup] selected={len(coord_paths)} device={device} "
        f"batch_size={args.batch_size} workers={args.workers} "
        f"shard={args.shard_index}/{args.num_shards}",
        flush=True,
    )

    failures = 0
    for index, coord_path in enumerate(coord_paths, start=1):
        slide_id = coord_path.stem
        output_path = output / "h5_files" / f"{slide_id}.h5"
        with h5py.File(coord_path, "r") as handle:
            expected_rows = len(handle["coords"])
        row = by_slide.get(slide_id, {"slide_id": slide_id})
        started = time.time()
        print(
            f"[slide] {index}/{len(coord_paths)} {slide_id} "
            f"patches={expected_rows}",
            flush=True,
        )
        try:
            rows_found, dimension = (
                (0, 0)
                if args.no_resume
                else validate_feature_file(output_path, expected_rows)
            )
            if (rows_found, dimension) == (expected_rows, 768):
                row["status"] = "already_completed"
            else:
                rows_found, dimension = extract_slide(
                    slide_path=slides[slide_id.lower()],
                    coord_path=coord_path,
                    output_path=output_path,
                    model=model,
                    device=device,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    clam_repo=args.clam_repo.resolve(),
                )
                row["status"] = "completed"
            row["patches"] = rows_found
            row["feature_dim"] = dimension
            row["output_path"] = str(output_path)
            row["error"] = ""
        except Exception as exc:
            failures += 1
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[error] {slide_id}: {row['error']}", flush=True)
            traceback.print_exc()
            if args.fail_fast:
                by_slide[slide_id] = row
                row["elapsed_seconds"] = round(time.time() - started, 3)
                row["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                atomic_write_csv(status_path, list(by_slide.values()))
                raise
        finally:
            row["elapsed_seconds"] = round(time.time() - started, 3)
            row["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            by_slide[slide_id] = row
            atomic_write_csv(status_path, list(by_slide.values()))

    completed = sum(
        row.get("status") in {"completed", "already_completed"}
        for row in by_slide.values()
    )
    print(
        f"[done] completed_total={completed} failures={failures}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
