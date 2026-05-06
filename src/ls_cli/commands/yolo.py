"""YOLO backend: deploy, status, export, visualize, train, metrics."""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import typer
from label_studio_sdk.converter.brush import decode_rle

from ls_cli.client import auth_headers, get_client
from ls_cli.config import Settings
from ls_cli.utils.dataset import export_yolo_dataset, visualize_yolo_dataset
from ls_cli.utils.docker import (
    compose_restart,
    container_inspect_env,
    container_is_running,
    docker_cp,
    docker_exec,
    docker_exec_python,
    resolve_backend_url,
)
from ls_cli.utils.metrics import KEY_HPARAMS, export_run, f1_from_pr, list_runs
from ls_cli.utils.output import console, error, info, print_table, success

app = typer.Typer(help="Manage the YOLO segmentation backend.")

CONTAINER_NAME = "yolo-seg"
WEIGHTS_DIR = "/data/weights"


@app.command()
def deploy(
    weights: Path = typer.Argument(help="Path to .pt weights file"),
    container: str = typer.Option(CONTAINER_NAME, help="Docker container name"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Deploy YOLO weights: copy into the volume and restart so the model picks them up."""
    settings = Settings.load(env_file, require_api_key=False)

    if not weights.is_file():
        error(f"Weights file not found: {weights}")
        raise typer.Exit(1)
    if weights.suffix != ".pt":
        error(f"Expected a .pt file, got: {weights.name}")
        raise typer.Exit(1)
    if not container_is_running(container):
        error(f"Container '{container}' is not running. Start it first (ls-cli up).")
        raise typer.Exit(1)

    dest = f"{WEIGHTS_DIR}/{weights.name}"
    info(f"Copying {weights} → {container}:{dest}")
    docker_exec(container, "mkdir", "-p", WEIGHTS_DIR)
    docker_cp(str(weights), f"{container}:{dest}")

    if not settings.compose_models_file:
        error("Could not locate the models compose file. Restart the container manually.")
        raise typer.Exit(1)

    info(f"Restarting {container}...")
    compose_restart(settings.compose_models_file, service=container, env_file=settings.env_file)
    success(f"Deployed {weights.name}. The backend will load it on the next prediction.")


@app.command()
def status(
    container: str = typer.Option(CONTAINER_NAME, help="Docker container name"),
) -> None:
    """Show the YOLO backend's current state."""
    running = container_is_running(container)
    info(f"Container:  {container}")
    info(f"Running:    {'yes' if running else 'no'}")
    if running:
        weights_dir = container_inspect_env(container, "YOLO_WEIGHTS_DIR") or WEIGHTS_DIR
        conf = container_inspect_env(container, "YOLO_CONF")
        info(f"Weights:    {weights_dir} (newest .pt loaded automatically)")
        info(f"Confidence: {conf or 'unknown'}")


@app.command()
def export(
    project_id: Optional[int] = typer.Option(None, help="Project ID (auto-resolved if only one)"),
    output_dir: Path = typer.Option(Path("exports/yolo_dataset"), help="Where to write the dataset"),
    split: float = typer.Option(0.8, help="Train fraction (rest goes to val)"),
    test_split: float = typer.Option(0.0, "--test-split", help="Fraction held out as a true test set (e.g. 0.1)"),
    seed: int = typer.Option(42, help="Random seed for the split"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Export annotated tasks as a YOLO segmentation dataset."""
    settings = Settings.load(env_file)
    ls = get_client(settings)
    pid = _resolve_project_id(ls, project_id)
    project = ls.projects.get(id=pid)
    info(f"Project: {project.title} (id={pid})")

    classes, exported, skipped = export_yolo_dataset(
        settings, ls, pid, output_dir, split=split, seed=seed, test_split=test_split,
    )
    info(f"Classes: {classes}")
    success(f"Exported {exported} tasks ({skipped} skipped) → {output_dir.resolve()}")
    if test_split > 0:
        n_test = (output_dir / "images" / "test")
        info(f"Test split: {len(list(n_test.glob('*')))} images in {n_test}")


@app.command()
def visualize(
    dataset_dir: Path = typer.Option(Path("exports/yolo_dataset"), help="Dataset directory"),
    split: str = typer.Option("all", help="Which split to render (train, val, all)"),
    max_images: Optional[int] = typer.Option(None, "--max-images", "-n", help="Stop after N images"),
    save_dir: Optional[Path] = typer.Option(None, help="Save PNGs here instead of showing interactively"),
) -> None:
    """Render image + polygon overlays for an exported dataset."""
    if split not in ("train", "val", "all"):
        error(f"--split must be one of: train, val, all (got {split})")
        raise typer.Exit(1)

    shown, issues = visualize_yolo_dataset(
        dataset_dir, split=split, max_images=max_images, save_dir=save_dir,
    )
    info(f"Visualised {shown} image(s).")
    if issues:
        info(f"\n{len(issues)} issue(s):")
        for msg in issues:
            info(f"  [!] {msg}")


@app.command()
def train(
    project_id: Optional[int] = typer.Option(None, help="Project ID (auto-resolved if only one)"),
    test_split: float = typer.Option(0.0, "--test-split", help="Fraction held out as test set, evaluated after training (e.g. 0.1)"),
    max_images: Optional[int] = typer.Option(None, "--max-images", help="Randomly sample this many images for training (uses all if unset)"),
    container: str = typer.Option(CONTAINER_NAME, help="YOLO container name"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Trigger training on the YOLO backend.

    Equivalent to clicking 'Start Training' in the Label Studio UI: LS sends a
    START_TRAINING webhook to the backend, which exports the dataset, trains,
    saves new weights, and reloads itself in-process.

    Pass --test-split 0.1 to hold out 10%% of tasks as a true test set.
    The backend will evaluate on them after training and save results to
    runs/yolo/<run>/test_metrics.json.

    Pass --max-images 200 to train on a random subset of 200 images.
    """
    settings = Settings.load(env_file)
    ls = get_client(settings)
    pid = _resolve_project_id(ls, project_id)

    backends = [b for b in ls.ml.list(project=pid) if (b.title or "").startswith("YOLO")]
    if not backends:
        error(f"No YOLO backend attached to project {pid}. Run: ls-cli backend switch {pid} yolo")
        raise typer.Exit(1)

    if test_split > 0 or max_images is not None:
        if not container_is_running(container):
            error(f"Container '{container}' is not running.")
            raise typer.Exit(1)
        cfg: dict = {}
        if test_split > 0:
            cfg["test_split"] = test_split
        if max_images is not None:
            cfg["max_images"] = max_images
        docker_exec_python(
            container,
            f"import json; open('/data/train_config.json','w').write(json.dumps({cfg}))",
        )
        if test_split > 0:
            info(f"Test split: {test_split:.0%} of tasks held out for post-training evaluation.")
        if max_images is not None:
            info(f"Max images: training on a random subset of {max_images} images.")

    backend = backends[0]
    info(f"Triggering training on '{backend.title}' (id={backend.id}) for project {pid}...")
    _start_training(settings, ls, backend.id)
    success("Training started. Check the YOLO container logs for progress.")
    if test_split > 0:
        info("Test evaluation runs automatically after training. Results → runs/yolo/<run>/test_metrics.json")


def _start_training(settings, ls, backend_id: int) -> None:
    """Start a training run via the LS API. Tries the SDK first, then a raw POST."""
    train_method = getattr(ls.ml, "train", None)
    if callable(train_method):
        try:
            train_method(id=backend_id)
            return
        except Exception:
            pass

    base = settings.label_studio_url.rstrip("/")
    resp = requests.post(
        f"{base}/api/ml/{backend_id}/train/",
        headers=auth_headers(ls),
        timeout=30,
    )
    resp.raise_for_status()


@app.command()
def evaluate(
    project_id: Optional[int] = typer.Option(None, help="Project ID (auto-resolved if only one)"),
    max_images: Optional[int] = typer.Option(
        None, "--max-images", "-n", help="Random sample of N annotated tasks (default: all)"
    ),
    seed: int = typer.Option(42, help="Random seed for --max-images sampling"),
    run: Optional[str] = typer.Option(None, "--run", "-r", help="Training run whose test_ids.json to use (default: latest run with a test split)"),
    export: Optional[Path] = typer.Option(None, "--export", help="Save results as JSON to this file or directory"),
    url: Optional[str] = typer.Option(None, "--url", help="Override the YOLO predict endpoint (e.g. http://localhost:9090)"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Evaluate YOLO prediction quality against ground truth annotations using IoU.

    For each annotated task, YOLO's predictions are compared to the human-verified
    brush annotations. Reports per-class and overall mean IoU.

    Use --url to override the YOLO endpoint when running the CLI from outside
    Docker (e.g. --url http://localhost:9090).
    """
    settings = Settings.load(env_file)
    ls = get_client(settings)
    pid = _resolve_project_id(ls, project_id)

    yolo_url = url.rstrip("/") if url else resolve_backend_url(settings.yolo_url, "yolo-seg")
    base = settings.label_studio_url.rstrip("/")

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
    info(f"YOLO endpoint: {yolo_url}")

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
        info("[WARN] No test split found — evaluating on all annotated tasks (comparison will be biased for YOLO).")

    if max_images is not None and len(tasks) > max_images:
        random.seed(seed)
        tasks = random.sample(tasks, max_images)

    try:
        requests.post(
            f"{yolo_url}/setup",
            json={
                "project": str(pid),
                "schema": label_config,
                "hostname": settings.label_studio_url,
                "access_token": settings.label_studio_api_key,
            },
            timeout=30,
        ).raise_for_status()
    except requests.RequestException as exc:
        error(f"Could not set up YOLO model: {exc}")
        raise typer.Exit(1)

    info(f"Evaluating {len(tasks)} task(s)...\n")

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
    failed = 0

    for task in tasks:
        task_id = task["id"]
        image_uri = task.get("data", {}).get("image", "")

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

        payload = {
            "tasks": [{"id": task_id, "data": {"image": image_uri}}],
            "project": str(pid),
            "label_config": label_config,
            "params": {},
        }
        try:
            pred_resp = requests.post(
                f"{yolo_url}/predict",
                json=payload,
                timeout=60,
            )
            pred_resp.raise_for_status()
            pred_data = pred_resp.json()
        except requests.RequestException as exc:
            info(f"  [WARN] task {task_id}: predict failed: {exc}")
            failed += 1
            continue

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

    def _mean(values: list[float]) -> str:
        return f"{sum(values) / len(values):.4f}" if values else "—"

    rows: list[tuple[str, str, str, str, str]] = []
    all_ious: list[float] = []
    all_dices: list[float] = []
    all_recalls: list[float] = []
    for cls in classes:
        rows.append((
            cls,
            _mean(class_ious[cls]),
            _mean(class_dices[cls]),
            _mean(class_recalls[cls]),
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
            str(len(all_ious)),
        ))

    print_table(
        f"YOLO evaluation — {len(tasks)} task(s), {len(classes)} class(es)",
        ["Class", "Mean IoU", "Dice", "Recall", "Tasks"],
        rows,
    )

    if export is not None:
        def _mean_f(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        data: dict = {
            "model": "yolo",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "project_id": pid,
            "run": run,
            "tasks_evaluated": len(tasks),
            "classes": {
                cls: {
                    "iou": _mean_f(class_ious[cls]),
                    "dice": _mean_f(class_dices[cls]),
                    "recall": _mean_f(class_recalls[cls]),
                    "n": len(class_ious[cls]),
                }
                for cls in classes
            },
            "overall": {
                "iou": _mean_f(all_ious),
                "dice": _mean_f(all_dices),
                "recall": _mean_f(all_recalls),
                "n": len(all_ious),
            },
        }
        out = Path(export)
        if out.is_dir() or not out.suffix:
            out.mkdir(parents=True, exist_ok=True)
            out = out / f"yolo_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
        success(f"Results saved → {out}")

    if failed:
        info(f"\n{failed} task(s) failed. Is YOLO running and reachable at {yolo_url}?")


@app.command()
def runs(
    sort_by: str = typer.Option(
        "mtime", help="Sort key: mtime | mAP50 | mAP50-95 | epochs"
    ),
) -> None:
    """List all training runs with key hyperparameters and final metrics."""
    all_runs = list_runs()
    if not all_runs:
        info("No training runs found in runs/yolo/. Run training first.")
        return

    sort_key_map = {
        "mtime": lambda r: r.path.stat().st_mtime,
        "mAP50": lambda r: r.best_metrics.get("metrics/mAP50(M)") or r.best_metrics.get("metrics/mAP50(B)") or 0,
        "mAP50-95": lambda r: r.best_metrics.get("metrics/mAP50-95(M)") or r.best_metrics.get("metrics/mAP50-95(B)") or 0,
        "epochs": lambda r: r.epochs_completed,
    }
    keyfn = sort_key_map.get(sort_by, sort_key_map["mtime"])
    all_runs.sort(key=keyfn, reverse=True)

    columns = ["Run", "Epochs", "imgsz", "batch", "Box mAP50", "Box mAP50-95", "Mask mAP50", "Mask mAP50-95"]
    rows = []
    for r in all_runs:
        rows.append((
            r.name,
            f"{r.epochs_completed}/{int(r.hparams.get('epochs', 0)) or '?'}",
            r.hparams.get("imgsz", "—"),
            r.hparams.get("batch", "—"),
            _fmt(r.best_metrics.get("metrics/mAP50(B)")),
            _fmt(r.best_metrics.get("metrics/mAP50-95(B)")),
            _fmt(r.best_metrics.get("metrics/mAP50(M)")),
            _fmt(r.best_metrics.get("metrics/mAP50-95(M)")),
        ))
    print_table(f"YOLO training runs ({len(all_runs)} total, sorted by {sort_by})", columns, rows)


@app.command()
def metrics(
    run: Optional[str] = typer.Option(None, "--run", "-r", help="Run name (default: latest)"),
    export: Optional[Path] = typer.Option(None, "--export", help="Copy plots + CSV + args to this dir"),
) -> None:
    """Show detailed metrics for a training run, or export them for a report."""
    runs_list = list_runs()
    if not runs_list:
        error("No training runs found in runs/yolo/. Run training first.")
        raise typer.Exit(1)

    if run:
        match = next((r for r in runs_list if r.name == run), None)
        if not match:
            error(f"No run named '{run}'. Try: ls-cli yolo runs")
            raise typer.Exit(1)
        target = match
    else:
        target = runs_list[0]

    info(f"Run: {target.name}")
    info(f"Path: {target.path}")
    info("")

    if target.hparams:
        rows = [(k, target.hparams.get(k, "—")) for k in KEY_HPARAMS]
        print_table("Hyperparameters", ["Key", "Value"], rows)
        info("")

    if target.best_metrics:
        bp = target.best_metrics.get("metrics/precision(B)") or 0
        br = target.best_metrics.get("metrics/recall(B)") or 0
        mp = target.best_metrics.get("metrics/precision(M)") or 0
        mr = target.best_metrics.get("metrics/recall(M)") or 0
        rows = [
            ("Box mAP50",      _fmt(target.best_metrics.get("metrics/mAP50(B)"))),
            ("Box mAP50-95",   _fmt(target.best_metrics.get("metrics/mAP50-95(B)"))),
            ("Box precision",  _fmt(bp)),
            ("Box recall",     _fmt(br)),
            ("Box F1",         _fmt(f1_from_pr(bp, br))),
            ("Mask mAP50",     _fmt(target.best_metrics.get("metrics/mAP50(M)"))),
            ("Mask mAP50-95",  _fmt(target.best_metrics.get("metrics/mAP50-95(M)"))),
            ("Mask precision", _fmt(mp)),
            ("Mask recall",    _fmt(mr)),
            ("Mask F1",        _fmt(f1_from_pr(mp, mr))),
        ]
        print_table(f"Best-epoch metrics ({target.epochs_completed}/{int(target.hparams.get('epochs', 0)) or '?'} epochs)", ["Metric", "Value"], rows)
        info("")
    else:
        info("No metrics yet (results.csv missing or empty).")

    if target.test_metrics:
        tp = target.test_metrics.get("metrics/precision(B)") or 0
        tr = target.test_metrics.get("metrics/recall(B)") or 0
        mp2 = target.test_metrics.get("metrics/precision(M)") or 0
        mr2 = target.test_metrics.get("metrics/recall(M)") or 0
        rows = [
            ("Box mAP50",      _fmt(target.test_metrics.get("metrics/mAP50(B)"))),
            ("Box mAP50-95",   _fmt(target.test_metrics.get("metrics/mAP50-95(B)"))),
            ("Box precision",  _fmt(tp)),
            ("Box recall",     _fmt(tr)),
            ("Box F1",         _fmt(f1_from_pr(tp, tr))),
            ("Mask mAP50",     _fmt(target.test_metrics.get("metrics/mAP50(M)"))),
            ("Mask mAP50-95",  _fmt(target.test_metrics.get("metrics/mAP50-95(M)"))),
            ("Mask precision", _fmt(mp2)),
            ("Mask recall",    _fmt(mr2)),
            ("Mask F1",        _fmt(f1_from_pr(mp2, mr2))),
        ]
        print_table("Test-set metrics (held-out)", ["Metric", "Value"], rows)
        info("")

        class_names = sorted({
            k.split("/")[1] for k in target.test_metrics if k.startswith("class/")
        })
        if class_names:
            cls_rows = []
            for cls_name in class_names:
                cls_rows.append((
                    cls_name,
                    _fmt(target.test_metrics.get(f"class/{cls_name}/box_mAP50")),
                    _fmt(target.test_metrics.get(f"class/{cls_name}/box_mAP50-95")),
                    _fmt(target.test_metrics.get(f"class/{cls_name}/mask_mAP50")),
                    _fmt(target.test_metrics.get(f"class/{cls_name}/mask_mAP50-95")),
                ))
            print_table(
                "Test-set per-class metrics",
                ["Class", "Box mAP50", "Box mAP50-95", "Mask mAP50", "Mask mAP50-95"],
                cls_rows,
            )
            info("")

    if target.has_plots:
        info(f"Plots available in {target.path} (results.png, confusion_matrix.png, P/R/F1/PR_curve.png, val_batch*.jpg)")
    else:
        info("No plots — was the run trained with plots=True?")

    if export:
        copied = export_run(target, export)
        success(f"Exported {len(copied)} files → {export}")
        info("Per-class metrics: see val output during training (printed to docker logs).")


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "—"
    return f"{f:.4f}"


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


def _resolve_project_id(ls, project_id: int | None) -> int:
    if project_id is not None:
        return project_id
    projects = list(ls.projects.list())
    if len(projects) == 1:
        return projects[0].id
    if not projects:
        error("No projects found. Run: ls-cli project create")
        raise typer.Exit(1)
    error("Multiple projects exist — pass --project-id:")
    for p in projects:
        error(f"  {p.id}: {p.title}")
    raise typer.Exit(1)
