#!/usr/bin/env python3
"""Convert CLAM HDF5 patch features to PIBD's tensor-file layout."""

import argparse
import csv
import os
from pathlib import Path

import h5py
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional CSV whose slide_id spelling is canonical for output names.",
    )
    return parser.parse_args()


def canonical_slide_stems(metadata_path: Path | None) -> dict[str, str]:
    if metadata_path is None:
        return {}

    stems: dict[str, str] = {}
    with metadata_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or "slide_id" not in reader.fieldnames:
            raise RuntimeError(f"slide_id column missing from {metadata_path}")
        for row in reader:
            stem = Path(row["slide_id"]).stem
            folded = stem.casefold()
            previous = stems.setdefault(folded, stem)
            if previous != stem:
                raise RuntimeError(
                    f"Ambiguous case-insensitive slide IDs: {previous!r}, {stem!r}"
                )
    return stems


def tensor_is_valid(path: Path, expected_rows: int, expected_dim: int) -> bool:
    if not path.is_file():
        return False
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.ndim == 2
        and tuple(tensor.shape) == (expected_rows, expected_dim)
        and tensor.dtype == torch.float32
    )


def main():
    args = parse_args()
    h5_paths = sorted(args.h5_dir.glob("*.h5"))
    if not h5_paths:
        raise RuntimeError(f"No HDF5 files found in {args.h5_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    canonical_stems = canonical_slide_stems(args.metadata)

    converted = 0
    reused = 0
    aliased = 0
    expected_outputs = []
    for index, h5_path in enumerate(h5_paths, start=1):
        output_stem = canonical_stems.get(h5_path.stem.casefold(), h5_path.stem)
        output_path = args.output_dir / f"{output_stem}.pt"
        legacy_path = args.output_dir / f"{h5_path.stem}.pt"
        expected_outputs.append(output_path)
        with h5py.File(h5_path, "r") as h5_file:
            dataset = h5_file["features"]
            rows, dim = dataset.shape
            if tensor_is_valid(output_path, rows, dim):
                reused += 1
            elif output_path != legacy_path and tensor_is_valid(
                legacy_path, rows, dim
            ):
                # Preserve the existing cache and expose the exact spelling used
                # by the metadata without duplicating a potentially large tensor.
                try:
                    os.link(legacy_path, output_path)
                except FileExistsError:
                    pass
                if not tensor_is_valid(output_path, rows, dim):
                    raise RuntimeError(f"Could not create valid alias {output_path}")
                aliased += 1
            else:
                features = torch.from_numpy(dataset[:]).float()
                temporary_path = output_path.with_suffix(
                    f".pt.tmp.{os.getpid()}"
                )
                torch.save(features, temporary_path)
                os.replace(temporary_path, output_path)
                converted += 1
        if index % 25 == 0 or index == len(h5_paths):
            print(
                f"[pibd-cache] {index}/{len(h5_paths)} "
                f"converted={converted} reused={reused} aliased={aliased}",
                flush=True,
            )

    missing_outputs = [path for path in expected_outputs if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(
            f"Feature cache incomplete: {len(missing_outputs)} canonical files missing"
        )


if __name__ == "__main__":
    main()
