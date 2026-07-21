#!/usr/bin/env python3
"""Reproduce the MRePath WSI preprocessing pipeline with legacy CLAM.

For every WSI this script performs:
  1. Otsu tissue segmentation on a downsampled slide.
  2. Non-overlapping 256 x 256 patch coordinate extraction at 20x.
  3. ImageNet-pretrained truncated ResNet50 feature extraction (1024-D).

Coordinates are stored in CLAM-compatible HDF5 files.  If a slide was scanned
at 40x, each 20x patch spans 512 x 512 level-0 pixels and is resized to
256 x 256 before entering ResNet50.
"""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_CLAM_DIR = ROOT / "third_party" / "CLAM"
DEFAULT_SOURCE = ROOT / "data" / "tcga_coadread" / "raw_svs"
DEFAULT_OUTPUT = ROOT / "data" / "tcga_coadread" / "clam_20x_resnet50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clam-dir", type=Path, default=DEFAULT_CLAM_DIR)
    parser.add_argument("--stages", default="segment,features",
                        help="Comma-separated stages: segment,features")
    parser.add_argument("--slide-id", default=None,
                        help="Only process a slide whose filename contains this value")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def add_clam_to_path(clam_dir: Path) -> None:
    if not (clam_dir / "wsi_core" / "WholeSlideImage.py").is_file():
        raise FileNotFoundError(f"CLAM not found at {clam_dir}")
    sys.path.insert(0, str(clam_dir))


def discover_slides(source: Path) -> list[Path]:
    slides = sorted(source.rglob("*.svs"), key=lambda p: (p.name.lower(), str(p)))
    if not slides:
        raise FileNotFoundError(f"No .svs files found recursively under {source}")
    duplicates: dict[str, list[Path]] = {}
    for slide in slides:
        duplicates.setdefault(slide.stem.lower(), []).append(slide)
    conflicts = {name: paths for name, paths in duplicates.items() if len(paths) > 1}
    if conflicts:
        sample = next(iter(conflicts.values()))
        raise RuntimeError(f"Duplicate slide basenames found: {sample}")
    return slides


def objective_power(slide: openslide.OpenSlide) -> tuple[float, str]:
    raw_power = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if raw_power:
        try:
            return float(raw_power), "objective-power"
        except ValueError:
            pass

    raw_mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if raw_mpp:
        try:
            # Standard brightfield relation: 0.25 um/px ~= 40x, 0.5 ~= 20x.
            return 10.0 / float(raw_mpp), "mpp-x"
        except (ValueError, ZeroDivisionError):
            pass

    # TCGA diagnostic slides are overwhelmingly 20x/40x.  A conservative 20x
    # fallback avoids accidentally doubling the physical field of view.
    return 20.0, "fallback-20x"


def select_patch_geometry(slide: openslide.OpenSlide) -> dict[str, Any]:
    power, power_source = objective_power(slide)
    target_downsample = power / 20.0
    downsamples = [float(value) for value in slide.level_downsamples]

    # Read from the highest pyramid level that is at least as detailed as 20x,
    # then resize to 256. This avoids using a 10x pyramid level on 40x scans.
    eligible = [idx for idx, value in enumerate(downsamples)
                if value <= target_downsample * 1.01]
    patch_level = max(eligible, key=lambda idx: downsamples[idx]) if eligible else 0
    level_downsample = downsamples[patch_level]
    scale = target_downsample / level_downsample
    read_patch_size = max(1, int(round(256 * scale)))
    effective_downsample = level_downsample * read_patch_size / 256.0

    return {
        "objective_power": power,
        "objective_source": power_source,
        "target_magnification": 20.0,
        "target_downsample": target_downsample,
        "patch_level": patch_level,
        "level_downsample": level_downsample,
        "read_patch_size": read_patch_size,
        "target_patch_size": 256,
        "effective_downsample": effective_downsample,
        "level_downsamples": downsamples,
    }


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, path)


def write_inventory(slides: list[Path], output: Path) -> list[dict[str, Any]]:
    inventory_path = output / "wsi_inventory.csv"
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(slides, start=1):
        try:
            with openslide.OpenSlide(str(path)) as slide:
                geometry = select_patch_geometry(slide)
                rows.append({
                    "slide_id": path.stem,
                    "filename": path.name,
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "width": slide.dimensions[0],
                    "height": slide.dimensions[1],
                    "level_count": slide.level_count,
                    **{key: value for key, value in geometry.items()
                       if key != "level_downsamples"},
                    "level_downsamples": json.dumps(geometry["level_downsamples"]),
                    "inventory_error": "",
                })
        except Exception as exc:  # keep a complete audit even if one slide is bad
            rows.append({
                "slide_id": path.stem,
                "filename": path.name,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "inventory_error": f"{type(exc).__name__}: {exc}",
            })
        if index % 25 == 0 or index == len(slides):
            print(f"[inventory] {index}/{len(slides)}", flush=True)
    atomic_write_csv(inventory_path, rows)
    return rows


def validate_coords(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with h5py.File(path, "r") as handle:
            coords = handle["coords"]
            if coords.ndim != 2 or coords.shape[1] != 2:
                return 0
            return int(coords.shape[0])
    except (OSError, KeyError):
        return 0


def validate_features(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    try:
        with h5py.File(path, "r") as handle:
            coords = handle["coords"]
            features = handle["features"]
            if coords.ndim != 2 or coords.shape[1] != 2 or features.ndim != 2:
                return 0, 0
            if coords.shape[0] != features.shape[0]:
                return 0, 0
            return int(features.shape[0]), int(features.shape[1])
    except (OSError, KeyError):
        return 0, 0


def segment_slide(slide_path: Path, output: Path) -> tuple[int, dict[str, Any]]:
    from wsi_core.WholeSlideImage import WholeSlideImage

    patch_dir = output / "patches"
    mask_dir = output / "masks"
    patch_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    wsi_object = WholeSlideImage(str(slide_path))
    slide = wsi_object.getOpenSlide()
    geometry = select_patch_geometry(slide)
    seg_level = slide.get_best_level_for_downsample(64)
    vis_level = seg_level
    seg_width, seg_height = slide.level_dimensions[seg_level]
    if seg_width * seg_height > 100_000_000:
        raise RuntimeError(f"Segmentation level is too large: {seg_width}x{seg_height}")

    wsi_object.segmentTissue(
        seg_level=seg_level,
        sthresh=8,
        mthresh=7,
        close=4,
        use_otsu=True,
        filter_params={"a_t": 16, "a_h": 4, "max_n_holes": 8},
        ref_patch_size=512,
        keep_ids=[],
        exclude_ids=[],
    )
    if not wsi_object.contours_tissue:
        raise RuntimeError("Otsu segmentation found no tissue contours")

    mask = wsi_object.visWSI(vis_level=vis_level, line_thickness=100)
    mask.save(mask_dir / f"{slide_path.stem}.jpg")
    wsi_object.process_contours(
        save_path=str(patch_dir),
        patch_level=geometry["patch_level"],
        patch_size=geometry["read_patch_size"],
        step_size=geometry["read_patch_size"],
        use_padding=True,
        contour_fn="four_pt",
    )
    try:
        slide.close()
    except Exception:
        pass

    coord_path = patch_dir / f"{slide_path.stem}.h5"
    count = validate_coords(coord_path)
    if count == 0:
        raise RuntimeError("Patch extraction produced no valid coordinates")
    return count, geometry


def collate_features(batch: list[tuple[torch.Tensor, np.ndarray]]) -> tuple[torch.Tensor, np.ndarray]:
    images = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return images, coords


def save_feature_batch(path: Path, features: np.ndarray, coords: np.ndarray, mode: str) -> None:
    with h5py.File(path, mode) as handle:
        for key, values in (("features", features), ("coords", coords)):
            if key not in handle:
                maxshape = (None,) + values.shape[1:]
                handle.create_dataset(key, data=values, maxshape=maxshape, chunks=True)
            else:
                dataset = handle[key]
                old_size = dataset.shape[0]
                dataset.resize(old_size + values.shape[0], axis=0)
                dataset[old_size:] = values


def build_model(device: torch.device) -> torch.nn.Module:
    from models.resnet_custom import resnet50_baseline

    model = resnet50_baseline(pretrained=True).to(device)
    model.eval()
    return model


def extract_features(
    slide_path: Path,
    output: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[int, int]:
    from datasets.dataset_h5 import Whole_Slide_Bag_FP

    coord_path = output / "patches" / f"{slide_path.stem}.h5"
    feature_dir = output / "h5_files"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / f"{slide_path.stem}.h5"
    temp_path = feature_path.with_suffix(".h5.partial")
    temp_path.unlink(missing_ok=True)

    with h5py.File(coord_path, "r") as handle:
        read_patch_size = int(handle["coords"].attrs["patch_size"])

    slide = openslide.OpenSlide(str(slide_path))
    dataset = Whole_Slide_Bag_FP(
        file_path=str(coord_path),
        wsi=slide,
        pretrained=True,
        target_patch_size=256 if read_patch_size != 256 else -1,
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
    with torch.inference_mode():
        for batch_index, (images, coords) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            features = model(images).cpu().numpy()
            save_feature_batch(temp_path, features, coords, mode)
            mode = "a"
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"[features] {slide_path.stem}: batch {batch_index}/{len(loader)}", flush=True)
    slide.close()
    os.replace(temp_path, feature_path)

    count, dimension = validate_features(feature_path)
    if count != len(dataset) or dimension != 1024:
        raise RuntimeError(
            f"Feature validation failed: got {count}x{dimension}, expected {len(dataset)}x1024"
        )
    return count, dimension


def main() -> int:
    args = parse_args()
    add_clam_to_path(args.clam_dir.resolve())
    stages = {value.strip() for value in args.stages.split(",") if value.strip()}
    unknown = stages - {"segment", "features"}
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    slides = discover_slides(source)
    inventory = write_inventory(slides, output)
    bad_inventory = [row for row in inventory if row.get("inventory_error")]
    if bad_inventory:
        print(f"[warning] {len(bad_inventory)} slides failed inventory inspection", flush=True)

    if args.slide_id:
        slides = [slide for slide in slides if args.slide_id.lower() in slide.name.lower()]
        if not slides:
            raise ValueError(f"No slide matched --slide-id {args.slide_id!r}")
    if args.limit is not None:
        slides = sorted(slides, key=lambda path: path.stat().st_size)[:args.limit]

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is not available")
    model = build_model(device) if "features" in stages else None
    print(f"[setup] slides={len(slides)} stages={sorted(stages)} device={device}", flush=True)

    status_path = output / "preprocess_status.csv"
    status_rows: list[dict[str, Any]] = []
    if status_path.is_file():
        status_rows = pd.read_csv(status_path).fillna("").to_dict("records")
    status_by_id = {str(row["slide_id"]): row for row in status_rows}

    failures = 0
    for index, slide_path in enumerate(slides, start=1):
        slide_id = slide_path.stem
        row = status_by_id.get(slide_id, {"slide_id": slide_id, "path": str(slide_path.resolve())})
        started = time.time()
        print(f"[slide] {index}/{len(slides)} {slide_id}", flush=True)
        try:
            coord_path = output / "patches" / f"{slide_id}.h5"
            coord_count = validate_coords(coord_path) if not args.no_resume else 0
            if "segment" in stages and coord_count == 0:
                coord_count, geometry = segment_slide(slide_path, output)
                row.update(geometry)
                row["segment_status"] = "completed"
            elif coord_count:
                row["segment_status"] = "already_completed"
            row["patch_count"] = coord_count

            feature_path = output / "h5_files" / f"{slide_id}.h5"
            feature_count, feature_dim = validate_features(feature_path) if not args.no_resume else (0, 0)
            if "features" in stages:
                if coord_count == 0:
                    raise RuntimeError("No valid patch coordinate file for feature extraction")
                if feature_count == 0 or feature_dim != 1024:
                    assert model is not None
                    feature_count, feature_dim = extract_features(
                        slide_path, output, model, device, args.batch_size, args.workers
                    )
                    row["feature_status"] = "completed"
                else:
                    row["feature_status"] = "already_completed"
            row["feature_count"] = feature_count
            row["feature_dim"] = feature_dim
            row["status"] = "completed"
            row["error"] = ""
        except Exception as exc:
            failures += 1
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[error] {slide_id}: {row['error']}", flush=True)
            traceback.print_exc()
            if args.fail_fast:
                status_by_id[slide_id] = row
                row["elapsed_seconds"] = round(time.time() - started, 3)
                atomic_write_csv(status_path, list(status_by_id.values()))
                raise
        finally:
            row["elapsed_seconds"] = round(time.time() - started, 3)
            row["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            status_by_id[slide_id] = row
            atomic_write_csv(status_path, list(status_by_id.values()))

    completed = sum(row.get("status") == "completed" for row in status_by_id.values())
    print(f"[done] selected={len(slides)} completed_total={completed} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
