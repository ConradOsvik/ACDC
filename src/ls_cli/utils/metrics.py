"""Read YOLO training run artifacts (results.csv, args.yaml, plots)."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PLOT_FILES = (
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "PR_curve.png",
    "MaskF1_curve.png",
    "MaskP_curve.png",
    "MaskR_curve.png",
    "MaskPR_curve.png",
    "labels.jpg",
    "labels_correlogram.jpg",
)

# Hyperparameters worth surfacing in the runs table.
KEY_HPARAMS = ("epochs", "imgsz", "batch", "lr0", "optimizer", "patience")

# Final-epoch metrics worth surfacing in the runs table.
KEY_METRICS = (
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
)


def _runs_dir() -> Path:
    """Repo's runs/yolo/ — bind-mounted to /data/runs in the yolo-seg container."""
    from ls_cli.config import _project_root

    root = _project_root()
    if not root:
        raise SystemExit("Could not locate project root.")
    return root / "runs" / "yolo"


@dataclass
class RunSummary:
    name: str
    path: Path
    hparams: dict[str, Any] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)
    best_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    test_ids: list[int] = field(default_factory=list)
    epochs_completed: int = 0
    has_plots: bool = False
    has_weights: bool = False


def list_runs() -> list[RunSummary]:
    """All training runs found in runs/yolo/, newest first (by mtime)."""
    base = _runs_dir()
    if not base.is_dir():
        return []

    runs: list[RunSummary] = []
    for path in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_dir():
            continue
        runs.append(load_run(path))
    return runs


def load_run(path: Path) -> RunSummary:
    summary = RunSummary(name=path.name, path=path)

    args_yaml = path / "args.yaml"
    if args_yaml.is_file():
        import yaml

        try:
            summary.hparams = yaml.safe_load(args_yaml.read_text()) or {}
        except Exception:
            summary.hparams = {}

    results_csv = path / "results.csv"
    if results_csv.is_file():
        rows = _read_results_csv(results_csv)
        if rows:
            summary.epochs_completed = len(rows)
            summary.final_metrics = rows[-1]
            summary.best_metrics = _pick_best_epoch(rows)

    test_json = path / "test_metrics.json"
    if test_json.is_file():
        import json
        try:
            summary.test_metrics = json.loads(test_json.read_text())
        except Exception:
            pass

    test_ids_json = path / "test_ids.json"
    if test_ids_json.is_file():
        import json
        try:
            summary.test_ids = json.loads(test_ids_json.read_text())
        except Exception:
            pass

    summary.has_plots = any((path / plot).is_file() for plot in PLOT_FILES)
    summary.has_weights = (path / "weights" / "best.pt").is_file()
    return summary


def _read_results_csv(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, float] = {}
            for k, v in raw.items():
                key = (k or "").strip()
                if not key:
                    continue
                try:
                    row[key] = float(v) if v not in (None, "") else float("nan")
                except (TypeError, ValueError):
                    pass
            if row:
                rows.append(row)
    return rows


def _pick_best_epoch(rows: list[dict[str, float]]) -> dict[str, float]:
    """Best row is the one with highest mAP50-95 (mask if present, else box)."""
    if not rows:
        return {}
    keyfn_priority = ("metrics/mAP50-95(M)", "metrics/mAP50-95(B)", "metrics/mAP50(B)")
    score_key = next((k for k in keyfn_priority if k in rows[0]), None)
    if not score_key:
        return rows[-1]
    return max(rows, key=lambda r: r.get(score_key, float("-inf")))


def f1_from_pr(p: float, r: float) -> float:
    if p + r <= 0:
        return 0.0
    return 2 * p * r / (p + r)


def export_run(run: RunSummary, dest: Path) -> list[Path]:
    """Copy plots + results.csv + args.yaml into dest. Returns list of copied files."""
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    for filename in (*PLOT_FILES, "results.csv", "args.yaml"):
        src = run.path / filename
        if src.is_file():
            target = dest / filename
            shutil.copy2(src, target)
            copied.append(target)

    # Optional: also copy any val_batch*.jpg (qualitative samples)
    for src in run.path.glob("val_batch*.jpg"):
        target = dest / src.name
        shutil.copy2(src, target)
        copied.append(target)

    return copied
