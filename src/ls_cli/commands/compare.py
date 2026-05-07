"""compare: side-by-side prediction visualizations for report figures."""

from __future__ import annotations

import io
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import typer
from PIL import Image, ImageDraw, ImageFont
from label_studio_sdk.converter.brush import decode_rle

from ls_cli.client import auth_headers, get_client
from ls_cli.config import Settings
from ls_cli.utils.docker import (
    container_inspect_env,
    container_is_running,
    docker_exec_python,
    resolve_backend_url,
)
from ls_cli.utils.metrics import list_runs
from ls_cli.utils.output import error, info, success

app = typer.Typer(help="Visualize and compare model predictions.")

_YOLO_CONTAINER = "yolo-seg"
_SAM3_CONTAINER = "sam3"
_PANEL_W = 400
_HEADER_H = 30
_GAP = 2
_ALPHA = 0.50

# BGR-style class colours (R, G, B)
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "Red Rust":   (220, 60,  40),
    "White Rust": (50,  130, 220),
}
_DEFAULT_COLOR: tuple[int, int, int] = (180, 180, 40)


@app.command()
def visualize(
    project_id: Optional[int] = typer.Option(None, help="Project ID (auto-resolved if only one)"),
    runs: list[str] = typer.Option([], "--run", "-r", help="YOLO run(s) to include (repeatable)"),
    include_sam3: bool = typer.Option(False, "--sam3", help="Include SAM3 predictions"),
    task_id: Optional[int] = typer.Option(None, "--task-id", "-t", help="Visualize a single specific task by ID"),
    max_images: Optional[int] = typer.Option(None, "--max-images", "-n", help="Limit to N tasks"),
    seed: int = typer.Option(42, help="Random seed for --max-images"),
    output_dir: Path = typer.Option(Path("exports/visualizations"), "--output", "-o"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Generate side-by-side prediction comparison images for a report.

    Produces one PNG per task showing the original image, ground truth, and
    one panel per model. Use --run (repeatable) to add YOLO runs and --sam3
    for SAM3.

    Example:
        ls-cli compare visualize --project-id 2 --sam3 \\
            --run train_20260506_210849 --run train_20260506_213149
    """
    if not runs and not include_sam3:
        error("Specify at least one model: --run <name> and/or --sam3")
        raise typer.Exit(1)

    settings = Settings.load(env_file)
    ls = get_client(settings)
    pid = _resolve_project_id(ls, project_id)
    base = settings.label_studio_url.rstrip("/")

    # Resolve YOLO runs
    run_summaries = []
    if runs:
        all_runs = list_runs()
        for run_name in runs:
            rs = next((r for r in all_runs if r.name == run_name), None)
            if not rs:
                error(f"No run named '{run_name}'. Try: ls-cli yolo runs")
                raise typer.Exit(1)
            if not rs.has_weights:
                error(f"Run '{run_name}' has no weights file.")
                raise typer.Exit(1)
            run_summaries.append(rs)

    if run_summaries and not container_is_running(_YOLO_CONTAINER):
        error(f"Container '{_YOLO_CONTAINER}' is not running.")
        raise typer.Exit(1)
    if include_sam3 and not container_is_running(_SAM3_CONTAINER):
        error(f"Container '{_SAM3_CONTAINER}' is not running.")
        raise typer.Exit(1)

    # Fetch project metadata
    try:
        proj_resp = requests.get(f"{base}/api/projects/{pid}/", headers=auth_headers(ls), timeout=30)
        proj_resp.raise_for_status()
    except requests.RequestException as exc:
        error(f"Could not fetch project: {exc}")
        raise typer.Exit(1)

    project_data = proj_resp.json()
    label_config: str = project_data.get("label_config", "")
    info(f"Project: {project_data.get('title', pid)} (id={pid})")

    classes: list[str] = []
    try:
        xml_root = ET.fromstring(label_config)
        for bl in xml_root.findall(".//BrushLabels"):
            for lbl in bl.findall("Label"):
                v = lbl.get("value")
                if v and v not in classes:
                    classes.append(v)
    except ET.ParseError as exc:
        error(f"Could not parse label config: {exc}")
        raise typer.Exit(1)

    if not classes:
        error("No BrushLabels found in project label config.")
        raise typer.Exit(1)
    info(f"Classes: {classes}")

    # Fetch annotated tasks
    try:
        export_resp = requests.get(
            f"{base}/api/projects/{pid}/export?exportType=JSON",
            headers=auth_headers(ls),
            timeout=120,
        )
        export_resp.raise_for_status()
    except requests.RequestException as exc:
        error(f"Could not export tasks: {exc}")
        raise typer.Exit(1)

    tasks = [t for t in export_resp.json() if t.get("annotations")]
    if not tasks:
        error("No annotated tasks found.")
        raise typer.Exit(1)

    if task_id is not None:
        tasks = [t for t in tasks if t["id"] == task_id]
        if not tasks:
            error(f"Task {task_id} not found or has no annotations.")
            raise typer.Exit(1)
    elif max_images is not None and len(tasks) > max_images:
        random.seed(seed)
        tasks = random.sample(tasks, max_images)

    info(f"Generating visualizations for {len(tasks)} task(s)...")

    # Build GT masks per task from annotations
    gt_by_task: dict[int, dict[str, np.ndarray | None]] = {}
    for task in tasks:
        task_id = task["id"]
        gt: dict[str, np.ndarray | None] = {cls: None for cls in classes}
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            for result in ann.get("result", []):
                if result.get("type") != "brushlabels":
                    continue
                value = result.get("value", {})
                rle = value.get("rle", [])
                label_names = value.get("brushlabels", [])
                if not rle or not label_names or label_names[0] not in classes:
                    continue
                cls_name = label_names[0]
                w = result.get("original_width", 100)
                h = result.get("original_height", 100)
                try:
                    m = _rle_to_mask(rle, w, h)
                    existing = gt[cls_name]
                    gt[cls_name] = m if existing is None else np.logical_or(existing, m).astype(np.uint8)
                except Exception:
                    pass
        gt_by_task[task_id] = gt

    # Collect predictions per model: list of (panel_label, masks_by_task)
    Masks = dict[str, np.ndarray | None]
    models: list[tuple[str, dict[int, Masks | None]]] = []

    # SAM3 via HTTP predict
    if include_sam3:
        sam3_url = resolve_backend_url(settings.sam3_url, _SAM3_CONTAINER)
        info(f"SAM3 endpoint: {sam3_url}")
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
            error(f"Could not set up SAM3: {exc}")
            raise typer.Exit(1)

        sam3_masks: dict[int, Masks | None] = {}
        for task in tasks:
            task_id = task["id"]
            image_uri = task.get("data", {}).get("image", "")
            try:
                pred_resp = requests.post(
                    f"{sam3_url}/predict",
                    json={
                        "tasks": [{"id": task_id, "data": {"image": image_uri}}],
                        "project": str(pid),
                        "label_config": label_config,
                        "params": {},
                    },
                    timeout=60,
                )
                pred_resp.raise_for_status()
                pred_data = pred_resp.json()
            except requests.RequestException as exc:
                info(f"  [WARN] SAM3 task {task_id}: {exc}")
                sam3_masks[task_id] = None
                continue

            pred: Masks = {cls: None for cls in classes}
            task_results = pred_data.get("results", [{}])
            for result in (task_results[0].get("result", []) if task_results else []):
                if result.get("type") != "brushlabels":
                    continue
                value = result.get("value", {})
                rle = value.get("rle", [])
                label_names = value.get("brushlabels", [])
                if not rle or not label_names or label_names[0] not in classes:
                    continue
                cls_name = label_names[0]
                w = result.get("original_width", 100)
                h = result.get("original_height", 100)
                try:
                    m = _rle_to_mask(rle, w, h)
                    existing = pred[cls_name]
                    pred[cls_name] = m if existing is None else np.logical_or(existing, m).astype(np.uint8)
                except Exception:
                    pass
            sam3_masks[task_id] = pred

        models.append(("SAM3", sam3_masks))

    # YOLO runs via docker exec
    label_set_lower = {c.lower() for c in classes}
    label_lower_to_cls = {c.lower(): c for c in classes}
    yolo_conf = container_inspect_env(_YOLO_CONTAINER, "YOLO_CONF") or "0.25" if run_summaries else "0.25"

    for rs in run_summaries:
        weights_path = f"/data/runs/{rs.name}/weights/best.pt"
        tasks_for_script = [
            {"id": t["id"], "image_uri": t.get("data", {}).get("image", "")}
            for t in tasks
        ]
        batch_script = f"""\
import sys, json
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/yolo-backend')
from ultralytics import YOLO
from PIL import Image

weights_path = {repr(weights_path)}
tasks_input = json.loads({repr(json.dumps(tasks_for_script))})
label_set_lower = {repr(label_set_lower)}
conf = float({repr(yolo_conf)})

model = YOLO(weights_path)
from utils import get_image_path

results = []
for t in tasks_input:
    task_id = t['id']
    image_uri = t['image_uri']
    try:
        local_path = get_image_path(image_uri, task_id=task_id)
        img = Image.open(local_path).convert('RGB')
        w, h = img.size
        preds = model.predict(img, conf=conf, verbose=False)
        preds_out = []
        if preds and preds[0].masks is not None:
            r = preds[0]
            for polygon_xy, cls_idx in zip(r.masks.xy, r.boxes.cls.cpu().numpy()):
                cls_name = r.names[int(cls_idx)]
                if label_set_lower and cls_name.lower() not in label_set_lower:
                    continue
                preds_out.append({{'class': cls_name, 'width': w, 'height': h, 'polygon': polygon_xy.tolist()}})
        results.append({{'task_id': task_id, 'predictions': preds_out}})
    except Exception as e:
        results.append({{'task_id': task_id, 'predictions': [], 'error': str(e)}})

print('PREDICTIONS:' + json.dumps(results))
"""
        info(f"Running YOLO run '{rs.name}'...")
        try:
            raw_output = docker_exec_python(_YOLO_CONTAINER, batch_script)
        except SystemExit as exc:
            error(f"YOLO run '{rs.name}' failed: {exc}")
            raise typer.Exit(1)

        pred_line = next((ln for ln in raw_output.splitlines() if ln.startswith("PREDICTIONS:")), None)
        if not pred_line:
            error(f"No prediction output for run '{rs.name}'.")
            raise typer.Exit(1)
        batch_predictions = json.loads(pred_line[len("PREDICTIONS:"):])

        yolo_masks: dict[int, Masks | None] = {}
        for item in batch_predictions:
            task_id = item["task_id"]
            if item.get("error"):
                info(f"  [WARN] task {task_id}: {item['error']}")
                yolo_masks[task_id] = None
                continue
            pred = {cls: None for cls in classes}
            for pred_item in item.get("predictions", []):
                cls_key = label_lower_to_cls.get(pred_item["class"].lower(), pred_item["class"])
                if cls_key not in pred:
                    continue
                try:
                    m = _polygon_to_mask(pred_item["polygon"], pred_item["width"], pred_item["height"])
                    existing = pred[cls_key]
                    pred[cls_key] = m if existing is None else np.logical_or(existing, m).astype(np.uint8)
                except Exception:
                    pass
            yolo_masks[task_id] = pred

        label = f"YOLO ({rs.images_trained_on} imgs)" if rs.images_trained_on is not None else "YOLO"
        models.append((label, yolo_masks))

    # Render and save one PNG per task
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for task in tasks:
        task_id = task["id"]
        image_uri = task.get("data", {}).get("image", "")

        orig = _download_image(image_uri, base, ls)
        if orig is None:
            info(f"  [WARN] task {task_id}: could not download image, skipping")
            continue

        panels = [
            _make_panel(orig, "Original"),
            _make_panel(_overlay(orig, gt_by_task[task_id]), "Ground Truth"),
        ]
        for label, masks_by_task in models:
            task_masks = masks_by_task.get(task_id)
            if task_masks is None:
                panels.append(_make_panel(_error_image(orig), f"{label} (failed)"))
            else:
                panels.append(_make_panel(_overlay(orig, task_masks), label))

        grid = _make_grid(panels)
        out_path = output_dir / f"task_{task_id:06d}.png"
        grid.save(out_path)
        saved += 1
        info(f"  Saved {out_path.name}")

    success(f"Saved {saved} image(s) → {output_dir.resolve()}")


# ── image helpers ──────────────────────────────────────────────────────────────

def _rle_to_mask(rle: list, width: int, height: int) -> np.ndarray:
    decoded = np.array(decode_rle(rle), dtype=np.uint8).reshape(height, width, 4)
    return (decoded[:, :, 3] > 0).astype(np.uint8)


def _polygon_to_mask(polygon: list, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon:
        pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def _overlay(image: Image.Image, masks: dict[str, np.ndarray | None]) -> Image.Image:
    """Alpha-blend per-class coloured masks onto the image."""
    img = np.array(image.convert("RGB"), dtype=np.float32)
    h, w = img.shape[:2]
    for cls_name, mask in masks.items():
        if mask is None or mask.sum() == 0:
            continue
        color = np.array(_CLASS_COLORS.get(cls_name, _DEFAULT_COLOR), dtype=np.float32)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        where = mask.astype(bool)
        img[where] = img[where] * (1 - _ALPHA) + color * _ALPHA
        # draw a 2-px outline for visibility
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(outline, contours, -1, 1, 2)
        img[outline.astype(bool)] = color
    return Image.fromarray(img.astype(np.uint8))


def _error_image(reference: Image.Image) -> Image.Image:
    img = reference.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), img.size], outline=(200, 50, 50), width=4)
    return img


def _make_panel(image: Image.Image, title: str) -> Image.Image:
    w, h = image.size
    new_h = int(h * _PANEL_W / w)
    resized = image.resize((_PANEL_W, new_h), Image.LANCZOS)
    panel = Image.new("RGB", (_PANEL_W, new_h + _HEADER_H), (30, 30, 30))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 6), title, fill=(240, 240, 240), font=font)
    panel.paste(resized, (0, _HEADER_H))
    return panel


def _make_grid(panels: list[Image.Image]) -> Image.Image:
    total_w = sum(p.width for p in panels) + _GAP * (len(panels) - 1)
    max_h = max(p.height for p in panels)
    grid = Image.new("RGB", (total_w, max_h), (50, 50, 50))
    x = 0
    for panel in panels:
        grid.paste(panel, (x, 0))
        x += panel.width + _GAP
    return grid


def _download_image(image_uri: str, base: str, ls) -> Image.Image | None:
    try:
        if image_uri.startswith("/data/"):
            resp = requests.get(f"{base}{image_uri}", headers=auth_headers(ls), timeout=30)
        else:
            resp = requests.get(image_uri, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


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
