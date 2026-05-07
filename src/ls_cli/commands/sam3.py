"""SAM3 backend: evaluate prediction quality against ground truth annotations."""

from __future__ import annotations

import json
import random
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import requests
from label_studio_sdk.converter.brush import decode_rle

import typer

from ls_cli.client import auth_headers, get_client
from ls_cli.config import Settings
from ls_cli.utils.docker import resolve_backend_url
from ls_cli.utils.metrics import list_runs
from ls_cli.utils.output import error, info, print_table, success

app = typer.Typer(help="Evaluate the SAM3 segmentation backend.")


@app.command()
def evaluate(
    project_id: Optional[int] = typer.Option(None, help="Project ID (auto-resolved if only one)"),
    max_images: Optional[int] = typer.Option(
        None, "--max-images", "-n", help="Random sample of N annotated tasks (default: all)"
    ),
    seed: int = typer.Option(42, help="Random seed for --max-images sampling"),
    run: Optional[str] = typer.Option(None, "--run", "-r", help="YOLO training run whose test_ids.json to use (default: latest run with a test split)"),
    export: Optional[Path] = typer.Option(None, "--export", help="Save results as JSON to this file or directory"),
    url: Optional[str] = typer.Option(None, "--url", help="Override the SAM3 predict endpoint (e.g. http://localhost:9091)"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Evaluate SAM3 prediction quality against ground truth annotations using IoU.

    For each annotated task, SAM3's text-mode predictions are compared to the
    human-verified brush annotations. Reports per-class and overall mean IoU.

    Use --url to override the SAM3 endpoint when running the CLI from outside
    Docker (e.g. --url http://localhost:9091).
    """

    settings = Settings.load(env_file)
    ls = get_client(settings)
    pid = _resolve_project_id(ls, project_id)

    sam3_url = url.rstrip("/") if url else resolve_backend_url(settings.sam3_url, "sam3")
    base = settings.label_studio_url.rstrip("/")

    # Fetch project metadata via raw API to get label_config reliably
    try:
        proj_resp = requests.get(
            f"{base}/api/projects/{pid}/",
            headers=auth_headers(ls),
            timeout=30,
        )
        proj_resp.raise_for_status()
    except requests.RequestException as exc:
        error(f"Could not fetch project from Label Studio: {exc}")
        raise typer.Exit(1)

    project_data = proj_resp.json()
    label_config: str = project_data.get("label_config", "")
    project_title: str = project_data.get("title", f"Project {pid}")

    info(f"Project: {project_title} (id={pid})")
    info(f"SAM3 endpoint: {sam3_url}")

    # Extract class names from label config
    classes: list[str] = []
    try:
        xml_root = ET.fromstring(label_config)
        for brush_labels in xml_root.findall(".//BrushLabels"):
            for label_el in brush_labels.findall("Label"):
                v = label_el.get("value")
                if v and v not in classes:
                    classes.append(v)
    except ET.ParseError as exc:
        error(f"Could not parse label config: {exc}")
        raise typer.Exit(1)

    if not classes:
        error("No BrushLabels found in the project label config.")
        raise typer.Exit(1)
    info(f"Classes: {classes}")

    # Fetch annotated tasks via the export endpoint
    try:
        export_resp = requests.get(
            f"{base}/api/projects/{pid}/export?exportType=JSON",
            headers=auth_headers(ls),
            timeout=120,
        )
        export_resp.raise_for_status()
    except requests.RequestException as exc:
        error(f"Could not export tasks from Label Studio: {exc}")
        raise typer.Exit(1)

    tasks = [t for t in export_resp.json() if t.get("annotations")]
    if not tasks:
        error("No annotated tasks found. Annotate some tasks first.")
        raise typer.Exit(1)

    test_ids = _load_test_ids(run)
    if test_ids is not None:
        id_set = set(test_ids)
        tasks = [t for t in tasks if t["id"] in id_set]
        if not tasks:
            error("None of the test-split tasks have annotations. Did you annotate the held-out images?")
            raise typer.Exit(1)
        info(f"Filtered to {len(tasks)} held-out test task(s) from run '{run or 'latest'}'.")
    else:
        info("[WARN] No test split found — pass --run to evaluate on a fair held-out set.")

    if max_images is not None and len(tasks) > max_images:
        random.seed(seed)
        tasks = random.sample(tasks, max_images)

    try:
        requests.post(
            f"{sam3_url}/setup",
            json={
                "project": str(pid),
                "schema": label_config,
                "hostname": settings.label_studio_url,
                "access_token": settings.label_studio_api_key,
            },
            timeout=30,
        ).raise_for_status()
    except requests.RequestException as exc:
        error(f"Could not set up SAM3 model: {exc}")
        raise typer.Exit(1)

    info(f"Evaluating {len(tasks)} task(s)...\n")
    eval_start = time.monotonic()

    def _rle_to_mask(rle: list, width: int, height: int) -> np.ndarray:
        decoded = np.array(decode_rle(rle), dtype=np.uint8).reshape(height, width, 4)
        return (decoded[:, :, 3] > 0).astype(np.uint8)

    def _mask_metrics(
        a: np.ndarray, b: np.ndarray
    ) -> tuple[float | None, float | None, float | None]:
        """Return (IoU, Dice, Recall) for a GT mask `a` and predicted mask `b`."""
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
        inter = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        gt_sum = float(a.sum())
        pred_sum = float(b.sum())
        iou = inter / union if union > 0 else None
        dice = (2 * inter) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else None
        recall = inter / gt_sum if gt_sum > 0 else None
        return iou, dice, recall

    class_ious: dict[str, list[float]] = {cls: [] for cls in classes}
    class_dices: dict[str, list[float]] = {cls: [] for cls in classes}
    class_recalls: dict[str, list[float]] = {cls: [] for cls in classes}
    class_tp50: dict[str, int] = {cls: 0 for cls in classes}
    class_fp50: dict[str, int] = {cls: 0 for cls in classes}
    class_fn50: dict[str, int] = {cls: 0 for cls in classes}
    failed = 0

    for task in tasks:
        task_id = task["id"]
        image_uri = task.get("data", {}).get("image", "")

        # Ground truth: union of brush annotations per class across all annotators
        gt_masks: dict[str, np.ndarray | None] = {cls: None for cls in classes}
        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue
            for result in annotation.get("result", []):
                if result.get("type") != "brushlabels":
                    continue
                value = result.get("value", {})
                rle = value.get("rle", [])
                label_names = value.get("brushlabels", [])
                if not rle or not label_names or label_names[0] not in classes:
                    continue
                cls = label_names[0]
                w = result.get("original_width", 100)
                h = result.get("original_height", 100)
                try:
                    m = _rle_to_mask(rle, w, h)
                    existing_gt = gt_masks[cls]
                    gt_masks[cls] = m if existing_gt is None else np.logical_or(existing_gt, m).astype(np.uint8)
                except Exception as exc:
                    info(f"  [WARN] task {task_id}: GT decode error: {exc}")

        # SAM3 predict — no context → text-prompt mode, queries all label names
        payload = {
            "tasks": [{"id": task_id, "data": {"image": image_uri}}],
            "project": str(pid),
            "label_config": label_config,
            "params": {},
        }
        try:
            pred_resp = requests.post(
                f"{sam3_url}/predict",
                json=payload,
                timeout=60,
            )
            pred_resp.raise_for_status()
            pred_data = pred_resp.json()
        except requests.RequestException as exc:
            info(f"  [WARN] task {task_id}: predict failed: {exc}")
            failed += 1
            continue

        # Predicted masks per class
        pred_masks: dict[str, np.ndarray | None] = {cls: None for cls in classes}
        task_results = pred_data.get("results", [{}])
        for result in (task_results[0].get("result", []) if task_results else []):
            if result.get("type") != "brushlabels":
                continue
            value = result.get("value", {})
            rle = value.get("rle", [])
            label_names = value.get("brushlabels", [])
            if not rle or not label_names or label_names[0] not in classes:
                continue
            cls = label_names[0]
            w = result.get("original_width", 100)
            h = result.get("original_height", 100)
            try:
                m = _rle_to_mask(rle, w, h)
                existing_pred = pred_masks[cls]
                pred_masks[cls] = m if existing_pred is None else np.logical_or(existing_pred, m).astype(np.uint8)
            except Exception as exc:
                info(f"  [WARN] task {task_id}: pred decode error: {exc}")

        # Metrics per class — skip only when both sides are empty for that class
        for cls in classes:
            gt = gt_masks[cls]
            pred = pred_masks[cls]
            if gt is None and pred is None:
                continue
            if gt is None:
                if pred is None:
                    continue
                gt_arr, pred_arr = np.zeros_like(pred), pred
            elif pred is None:
                gt_arr, pred_arr = gt, np.zeros_like(gt)
            else:
                gt_arr, pred_arr = gt, pred
            iou, dice, recall = _mask_metrics(gt_arr, pred_arr)
            if iou is not None:
                class_ious[cls].append(iou)
            if dice is not None:
                class_dices[cls].append(dice)
            if recall is not None:
                class_recalls[cls].append(recall)
            gt_exists = gt is not None and gt.sum() > 0
            pred_exists = pred is not None and pred.sum() > 0
            if gt_exists and pred_exists and iou is not None and iou >= 0.5:
                class_tp50[cls] += 1
            else:
                if pred_exists:
                    class_fp50[cls] += 1
                if gt_exists:
                    class_fn50[cls] += 1

    # Report
    def _mean(values: list[float]) -> str:
        return f"{sum(values) / len(values):.4f}" if values else "—"

    rows: list[tuple] = []
    all_ious: list[float] = []
    all_dices: list[float] = []
    all_recalls: list[float] = []
    all_ap50: list[float] = []
    all_rec50: list[float] = []
    for cls in classes:
        tp, fp, fn = class_tp50[cls], class_fp50[cls], class_fn50[cls]
        ap50 = tp / (tp + fp) if (tp + fp) > 0 else None
        rec50 = tp / (tp + fn) if (tp + fn) > 0 else None
        if ap50 is not None:
            all_ap50.append(ap50)
        if rec50 is not None:
            all_rec50.append(rec50)
        rows.append((
            cls,
            _mean(class_ious[cls]),
            _mean(class_dices[cls]),
            _mean(class_recalls[cls]),
            f"{ap50:.4f}" if ap50 is not None else "—",
            f"{rec50:.4f}" if rec50 is not None else "—",
            str(len(class_ious[cls])),
        ))
        all_ious.extend(class_ious[cls])
        all_dices.extend(class_dices[cls])
        all_recalls.extend(class_recalls[cls])

    if all_ious:
        rows.append((
            "OVERALL",
            _mean(all_ious),
            _mean(all_dices),
            _mean(all_recalls),
            _mean(all_ap50),
            _mean(all_rec50),
            str(len(all_ious)),
        ))

    elapsed = time.monotonic() - eval_start
    print_table(
        f"SAM3 evaluation — {len(tasks)} task(s), {len(classes)} class(es), {_fmt_duration(elapsed)}",
        ["Class", "Mean IoU", "Dice", "Recall", "mAP@50", "Recall@50", "Tasks"],
        rows,
    )

    if export is not None:
        def _mean_f(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        data: dict = {
            "model": "sam3",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "project_id": pid,
            "run": run,
            "tasks_evaluated": len(tasks),
            "duration_seconds": round(elapsed, 2),
            "classes": {
                cls: {
                    "iou": _mean_f(class_ious[cls]),
                    "dice": _mean_f(class_dices[cls]),
                    "recall": _mean_f(class_recalls[cls]),
                    "ap50": class_tp50[cls] / (class_tp50[cls] + class_fp50[cls]) if (class_tp50[cls] + class_fp50[cls]) > 0 else None,
                    "recall50": class_tp50[cls] / (class_tp50[cls] + class_fn50[cls]) if (class_tp50[cls] + class_fn50[cls]) > 0 else None,
                    "n": len(class_ious[cls]),
                }
                for cls in classes
            },
            "overall": {
                "iou": _mean_f(all_ious),
                "dice": _mean_f(all_dices),
                "recall": _mean_f(all_recalls),
                "ap50": _mean_f(all_ap50),
                "recall50": _mean_f(all_rec50),
                "n": len(all_ious),
            },
        }
        out = Path(export)
        if out.is_dir() or not out.suffix:
            out.mkdir(parents=True, exist_ok=True)
            out = out / f"sam3_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
        success(f"Results saved → {out}")

    if failed:
        info(f"\n{failed} task(s) failed. Is SAM3 running and reachable at {sam3_url}?")


def _load_test_ids(run: str | None) -> list[int] | None:
    """Return the held-out task IDs from a training run's test_ids.json.

    Looks up runs/yolo/<run>/test_ids.json, defaulting to the latest run
    that has one. Returns None if no test split is found.
    """
    all_runs = list_runs()
    if not all_runs:
        return None
    if run:
        match = next((r for r in all_runs if r.name == run), None)
        if not match:
            error(f"No run named '{run}'. Try: ls-cli yolo runs")
            raise typer.Exit(1)
        target = match
    else:
        target = next((r for r in all_runs if r.test_ids), None)
        if target is None:
            return None
    return target.test_ids or None


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _resolve_project_id(ls, project_id: int | None) -> int:
    if project_id is not None:
        return project_id
    projects = list(ls.projects.list())
    if len(projects) == 1:
        return projects[0].id
    if not projects:
        error("No projects found.")
        raise typer.Exit(1)
    error("Multiple projects exist — pass --project-id:")
    for p in projects:
        error(f"  {p.id}: {p.title}")
    raise typer.Exit(1)
