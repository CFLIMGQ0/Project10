"""Structured diagnostics for PC-CMKA-DDKAC experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _cpu_array(value: torch.Tensor) -> np.ndarray:
    return torch.as_tensor(value).detach().float().cpu().numpy()


class PCCMKADiagnosticRecorder:
    """Collect per-patient tensors and export JSONL, CSV and compressed NPZ."""

    def __init__(self, output_dir: str | Path, fold: int) -> None:
        self.output_dir = Path(output_dir)
        self.fold = int(fold)
        self.records: list[dict[str, Any]] = []
        self.arrays: dict[str, list[np.ndarray]] = {}

    def add(
        self,
        case_id: str,
        phase: str,
        diagnostics: dict[str, torch.Tensor],
        edge_diagnostics: list[dict[str, torch.Tensor]],
        *,
        epoch: int,
    ) -> None:
        if phase not in {"train", "val", "test"}:
            raise ValueError("phase must be train, val, or test")
        patient_index = 0
        row: dict[str, Any] = {
            "case_id": str(case_id),
            "fold": self.fold,
            "epoch": int(epoch),
            "phase": phase,
        }
        scalar_names = (
            "forward_seconds",
            "shared_lambda",
        )
        for name in scalar_names:
            if name in diagnostics:
                row[name] = float(_cpu_array(diagnostics[name]).reshape(-1)[0])
        for name in (
            "rho",
            "coefficient_norm",
            "infinity_scale",
            "krylov_scale",
            "krylov_error",
            "gates",
        ):
            if name in diagnostics:
                values = _cpu_array(diagnostics[name])[patient_index]
                for group, value in enumerate(values.reshape(-1)):
                    row[f"{name}_group_{group}"] = float(value)
        for name, value in diagnostics.items():
            array = _cpu_array(value)
            if array.ndim > 0 and array.shape[0] == 1:
                array = array[patient_index]
            key = f"record_{len(self.records):08d}_{name}"
            self.arrays.setdefault(key, []).append(array)
        for group, edge_data in enumerate(edge_diagnostics):
            for name, value in edge_data.items():
                key = f"record_{len(self.records):08d}_group_{group}_{name}"
                self.arrays.setdefault(key, []).append(
                    _cpu_array(value)[patient_index]
                )
        self.records.append(row)

    def save_fold_summary(
        self,
        *,
        c_index: float,
        best_epoch: int,
        elapsed_seconds: float,
        peak_gpu_bytes: int,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "fold": self.fold,
            "c_index": float(c_index),
            "best_epoch": int(best_epoch),
            "elapsed_seconds": float(elapsed_seconds),
            "peak_gpu_bytes": int(peak_gpu_bytes),
            "patient_records": len(self.records),
        }
        (self.output_dir / f"fold_{self.fold}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    def flush(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = self.output_dir / f"fold_{self.fold}_patients.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        csv_path = self.output_dir / f"fold_{self.fold}_patients.csv"
        if self.records:
            fieldnames = sorted(
                {key for record in self.records for key in record}
            )
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.records)
        np.savez_compressed(
            self.output_dir / f"fold_{self.fold}_diagnostics.npz",
            **{
                key: np.stack(values) if len(values) > 1 else values[0]
                for key, values in self.arrays.items()
            },
        )
