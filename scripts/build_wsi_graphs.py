#!/usr/bin/env python3
"""Build MRePath-compatible PyG graphs from CLAM feature HDF5 files.

This is a resumable, atomic wrapper around ``extract_graph.pt2graph``.  It
preserves the repository's graph algorithm (NMSLIB HNSW, radius=9) while
adding per-slide validation and a status CSV.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "tcga_coadread" / "clam_20x_resnet50"
DEFAULT_METADATA = ROOT / "datasets_csv" / "metadata" / "tcga_coadread.csv"
sys.path.insert(0, str(ROOT))

from extract_graph import pt2graph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", type=Path, default=DEFAULT_DATA_ROOT / "h5_files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_ROOT / "graph_files")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--radius", type=int, default=9)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--spatial-space", default="l2")
    parser.add_argument("--feature-space", default="cosinesimil")
    parser.add_argument("--slide-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def canonical_slide_names(metadata_csv: Path) -> dict[str, str]:
    if not metadata_csv.is_file():
        return {}
    metadata = pd.read_csv(metadata_csv)
    if "slide_id" not in metadata:
        raise ValueError(f"Missing slide_id column in {metadata_csv}")
    names = [Path(str(value)).stem for value in metadata["slide_id"].dropna()]
    mapping = {name.lower(): name for name in names}
    if len(mapping) != len(set(names)):
        raise ValueError("Metadata contains slide IDs that differ only by case")
    return mapping


def inspect_h5(path: Path, expected_feature_dim: int) -> tuple[int, int]:
    with h5py.File(path, "r") as handle:
        features = handle["features"]
        coords = handle["coords"]
        if features.ndim != 2 or features.shape[1] != expected_feature_dim:
            raise ValueError(f"Invalid features shape: {features.shape}")
        if coords.shape != (features.shape[0], 2):
            raise ValueError(f"Invalid coords shape: {coords.shape}")
        if features.shape[0] < 9:
            raise ValueError(f"Graph radius 9 requires at least 9 patches, got {features.shape[0]}")
        return int(features.shape[0]), int(features.shape[1])


def validate_graph(
    path: Path,
    expected_nodes: int,
    expected_feature_dim: int,
    radius: int,
    spatial_space: str,
    feature_space: str,
) -> dict[str, int]:
    # PyTorch >=2.6 defaults to weights_only=True. PyG Data objects are trusted
    # local artifacts here and therefore need an explicit full pickle load.
    graph = torch.load(path, map_location="cpu", weights_only=False)
    expected_edges = expected_nodes * (radius - 1)
    expected = {
        "x": (expected_nodes, expected_feature_dim),
        "centroid": (expected_nodes, 2),
        "edge_index": (2, expected_edges),
        "edge_latent": (2, expected_edges),
    }
    for key, shape in expected.items():
        value = getattr(graph, key, None)
        if value is None or tuple(value.shape) != shape:
            actual = None if value is None else tuple(value.shape)
            raise ValueError(f"Invalid {key} shape: {actual}, expected {shape}")
    for key in ("edge_index", "edge_latent"):
        edges = getattr(graph, key)
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= expected_nodes):
            raise ValueError(f"{key} contains an out-of-range node index")
    if not torch.isfinite(graph.x).all():
        raise ValueError("Graph features contain NaN or infinity")
    if getattr(graph, "spatial_metric", None) != spatial_space:
        raise ValueError("Graph spatial metric does not match the request")
    if getattr(graph, "feature_metric", None) != feature_space:
        raise ValueError("Graph feature metric does not match the request")
    if int(getattr(graph, "hyperedge_size", -1)) != radius:
        raise ValueError("Graph hyperedge size does not match the requested radius")
    return {"nodes": expected_nodes, "spatial_edges": expected_edges,
            "latent_edges": expected_edges, "spatial_metric": spatial_space,
            "feature_metric": feature_space, "hyperedge_size": radius}


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temp, index=False)
    os.replace(temp, path)


def main() -> int:
    args = parse_args()
    h5_dir = args.h5_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "graph_status.csv"

    h5_files = sorted(h5_dir.glob("*.h5"), key=lambda path: path.name.lower())
    if args.slide_id:
        h5_files = [path for path in h5_files if args.slide_id.lower() in path.name.lower()]
    if args.limit is not None:
        h5_files = sorted(h5_files, key=lambda path: path.stat().st_size)[:args.limit]
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files selected from {h5_dir}")

    canonical = canonical_slide_names(args.metadata_csv.resolve())
    destinations = [canonical.get(path.stem.lower(), path.stem) for path in h5_files]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Multiple HDF5 files map to the same graph filename")

    rows: list[dict[str, Any]] = []
    if status_path.is_file():
        rows = pd.read_csv(status_path).fillna("").to_dict("records")
    by_source = {str(row["source_h5"]): row for row in rows if row.get("source_h5")}

    failures = 0
    print(
        f"[setup] selected={len(h5_files)} radius={args.radius} "
        f"spatial={args.spatial_space} feature={args.feature_space} "
        f"output={output_dir}",
        flush=True,
    )
    for index, (h5_path, graph_stem) in enumerate(zip(h5_files, destinations), start=1):
        graph_path = output_dir / f"{graph_stem}.pt"
        temp_path = output_dir / f"{graph_stem}.pt.partial"
        source_key = str(h5_path.resolve())
        row = by_source.get(source_key, {"source_h5": source_key})
        started = time.time()
        print(f"[graph] {index}/{len(h5_files)} {h5_path.stem}", flush=True)
        try:
            nodes, feature_dim = inspect_h5(h5_path, args.feature_dim)
            existing_complete = (
                not args.no_resume
                and graph_path.is_file()
                and graph_path.stat().st_size > 0
                and row.get("status") == "completed"
            )
            if existing_complete and not args.verify_existing:
                row["status"] = "already_completed"
                row["nodes"] = nodes
                row["feature_dim"] = feature_dim
            else:
                if existing_complete:
                    stats = validate_graph(
                        graph_path,
                        nodes,
                        feature_dim,
                        args.radius,
                        args.spatial_space,
                        args.feature_space,
                    )
                else:
                    temp_path.unlink(missing_ok=True)
                    with h5py.File(h5_path, "r") as handle:
                        graph = pt2graph(
                            handle,
                            radius=args.radius,
                            spatial_space=args.spatial_space,
                            feature_space=args.feature_space,
                        )
                    torch.save(graph, temp_path)
                    stats = validate_graph(
                        temp_path,
                        nodes,
                        feature_dim,
                        args.radius,
                        args.spatial_space,
                        args.feature_space,
                    )
                    os.replace(temp_path, graph_path)
                    del graph
                row.update(stats)
                row["feature_dim"] = feature_dim
                row["status"] = "completed"
            row["graph_path"] = str(graph_path)
            row["graph_bytes"] = graph_path.stat().st_size
            row["error"] = ""
        except Exception as exc:
            failures += 1
            temp_path.unlink(missing_ok=True)
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[error] {h5_path.name}: {row['error']}", flush=True)
            traceback.print_exc()
            if args.fail_fast:
                row["elapsed_seconds"] = round(time.time() - started, 3)
                row["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                by_source[source_key] = row
                atomic_write_csv(status_path, list(by_source.values()))
                raise
        finally:
            row["elapsed_seconds"] = round(time.time() - started, 3)
            row["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            by_source[source_key] = row
            atomic_write_csv(status_path, list(by_source.values()))

    completed = sum(row.get("status") in {"completed", "already_completed"}
                    for row in by_source.values())
    print(f"[done] completed_total={completed} failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
