#!/usr/bin/env python3
"""Build persistent topology and feature hyperedge caches for WSI graphs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.hypergraph_cache import (
    build_cache_record,
    cache_matches,
    save_cache_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def build_one(source: Path, cache_dir: Path) -> tuple[str, str, int]:
    destination = cache_dir / source.name
    if destination.is_file():
        record = torch.load(
            destination, map_location="cpu", weights_only=False, mmap=True
        )
        if cache_matches(record, source):
            return source.name, "cached", destination.stat().st_size

    graph = torch.load(
        source, map_location="cpu", weights_only=False, mmap=True
    )
    record = build_cache_record(graph, source)
    save_cache_record(record, destination)
    return source.name, "built", destination.stat().st_size


def main() -> int:
    args = parse_args()
    sources = sorted(args.graph_dir.glob("*.pt"))
    if not sources:
        raise FileNotFoundError(f"No .pt graph files in {args.graph_dir}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    counts = {"built": 0, "cached": 0}
    total_bytes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(build_one, source, args.cache_dir): source
            for source in sources
        }
        for index, future in enumerate(as_completed(futures), start=1):
            name, status, size = future.result()
            counts[status] += 1
            total_bytes += size
            print(
                f"[{index}/{len(sources)}] {status}: {name} "
                f"({size / 2**20:.2f} MiB)",
                flush=True,
            )

    print(
        f"[complete] built={counts['built']} cached={counts['cached']} "
        f"total={total_bytes / 2**30:.2f} GiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
