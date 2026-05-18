"""Dataset statistics: analyze annotated tasks from Label Studio."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import typer
from label_studio_sdk.converter.brush import decode_rle
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ls_cli.client import auth_headers, get_client
from ls_cli.config import Settings
from ls_cli.utils.output import console, error, info, success

app = typer.Typer(help="Analyze dataset statistics.")


class ExportFormat(str, Enum):
    json = "JSON"
    csv = "CSV"
    tsv = "TSV"
    coco = "COCO"
    yolo = "YOLO"
    brush_to_coco = "BRUSH_TO_COCO"


@app.command()
def export(
    output: Path = typer.Argument(help="Output file, or directory when --with-images is set"),
    format: ExportFormat = typer.Option(ExportFormat.json, "--format", "-f", help="Export format"),
    with_images: bool = typer.Option(False, "--with-images", help="Download images to <output>/images/ (JSON format only)"),
    project_id: Optional[int] = typer.Option(None, help="Project ID (auto-resolved if only one)"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Export all annotated tasks from a Label Studio project to a file.

    With --with-images the output becomes a directory containing
    annotations.json and an images/ subfolder with all task images.
    The annotations.json paths are rewritten to relative image paths.
    """
    if with_images and format != ExportFormat.json:
        error("--with-images is only supported with --format json")
        raise typer.Exit(1)

    settings = Settings.load(env_file)
    ls = get_client(settings)
    pid = _resolve_project_id(ls, project_id)
    base = settings.label_studio_url.rstrip("/")

    default_extensions = {
        ExportFormat.json: ".json",
        ExportFormat.csv: ".csv",
        ExportFormat.tsv: ".tsv",
        ExportFormat.coco: ".json",
        ExportFormat.yolo: ".zip",
        ExportFormat.brush_to_coco: ".json",
    }

    info(f"Exporting project {pid} as {format.value}…")

    try:
        resp = requests.get(
            f"{base}/api/projects/{pid}/export",
            params={"exportType": format.value},
            headers=auth_headers(ls),
            timeout=300,
            stream=True,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        error(f"Export failed: {exc}")
        raise typer.Exit(1)

    if not with_images:
        out_path = output if output.suffix else output.with_suffix(default_extensions[format])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            dl_task = progress.add_task("Downloading…", total=None)
            with out_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
            progress.update(dl_task, description="Done.")
        size_kb = out_path.stat().st_size / 1024
        success(f"Saved {size_kb:,.1f} KB → {out_path.resolve()}")
        return

    # --- with-images path ---
    import base64

    out_dir = output
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = out_dir / "annotations.json"

    tasks = resp.json()
    headers = auth_headers(ls)

    failed = 0
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        dl_task = progress.add_task(f"Downloading 0/{len(tasks)} images…", total=None)

        for i, task in enumerate(tasks):
            image_uri: str = task.get("data", {}).get("image", "")
            if not image_uri:
                continue

            ext = Path(image_uri.split("?")[0]).suffix or ".jpg"
            img_name = f"task_{task['id']:06d}{ext}"
            img_dest = images_dir / img_name

            try:
                if image_uri.startswith(("s3://", "gs://", "azure://")):
                    fileuri_b64 = base64.b64encode(image_uri.encode()).decode()
                    img_resp = requests.get(
                        f"{base}/tasks/{task['id']}/resolve/",
                        params={"fileuri": fileuri_b64},
                        headers=headers,
                        timeout=60,
                        allow_redirects=True,
                    )
                elif image_uri.startswith("/data/"):
                    img_resp = requests.get(f"{base}{image_uri}", headers=headers, timeout=60)
                else:
                    img_resp = requests.get(image_uri, timeout=60)
                img_resp.raise_for_status()
                img_dest.write_bytes(img_resp.content)
                task["data"]["image"] = f"images/{img_name}"
            except Exception as exc:
                error(f"  task {task['id']}: image download failed: {exc}")
                failed += 1

            progress.update(dl_task, description=f"Downloading {i + 1}/{len(tasks)} images…")

        progress.update(dl_task, description="Done.")

    annotations_path.write_text(json.dumps(tasks, indent=2))

    total_images = len(tasks) - failed
    success(f"Saved {total_images} images + annotations.json → {out_dir.resolve()}")
    if failed:
        info(f"  ({failed} image(s) failed to download)")


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


def _rle_to_mask(rle: list, width: int, height: int) -> np.ndarray:
    decoded = np.array(decode_rle(rle), dtype=np.uint8).reshape(height, width, 4)
    return (decoded[:, :, 3] > 0).astype(np.uint8)


@app.command()
def stats(
    project_id: Optional[int] = typer.Option(None, help="Project ID (auto-resolved if only one)"),
    export: Optional[Path] = typer.Option(None, "--export", help="Save results as JSON to this file"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Compute dataset statistics from annotated tasks in Label Studio.

    Reports per-class image counts, annotation density, mask area distribution,
    and co-occurrence between classes. Useful for characterising dataset balance
    and coverage for a thesis or report.
    """
    settings = Settings.load(env_file)
    ls = get_client(settings)
    pid = _resolve_project_id(ls, project_id)
    base = settings.label_studio_url.rstrip("/")

    try:
        proj_resp = requests.get(
            f"{base}/api/projects/{pid}/",
            headers=auth_headers(ls),
            timeout=30,
        )
        proj_resp.raise_for_status()
    except requests.RequestException as exc:
        error(f"Could not fetch project: {exc}")
        raise typer.Exit(1)

    project_data = proj_resp.json()
    label_config: str = project_data.get("label_config", "")
    project_title: str = project_data.get("title", f"Project {pid}")

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

    info(f"Project: {project_title} (id={pid})")
    info(f"Classes: {classes}\n")

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

    all_tasks = export_resp.json()
    total_tasks = len(all_tasks)

    # --- decode masks and collect per-task stats ---

    # per_class: list of mask pixel areas for each instance
    per_class_areas: dict[str, list[int]] = {c: [] for c in classes}
    # per_class: list of per-image coverage fractions
    per_class_coverage: dict[str, list[float]] = {c: [] for c in classes}
    # per_class: images containing this class
    per_class_image_count: dict[str, int] = {c: 0 for c in classes}

    annotated_tasks: list[dict] = []
    unannotated_count = 0
    instances_per_task: list[int] = []  # total brush instances per annotated task
    classes_per_task: list[set[str]] = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task_progress = progress.add_task(f"Decoding masks for {total_tasks} tasks…", total=None)

        for task in all_tasks:
            annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
            brush_results = [
                r for a in annotations for r in a.get("result", [])
                if r.get("type") == "brushlabels"
            ]

            if not brush_results:
                unannotated_count += 1
                continue

            annotated_tasks.append(task)
            task_classes: set[str] = set()
            instance_count = 0

            for r in brush_results:
                value = r.get("value", {})
                rle = value.get("rle", [])
                label_names = value.get("brushlabels", [])
                if not rle or not label_names or label_names[0] not in classes:
                    continue

                cls = label_names[0]
                orig_w = r.get("original_width", 0)
                orig_h = r.get("original_height", 0)
                if orig_w <= 0 or orig_h <= 0:
                    continue

                try:
                    mask = _rle_to_mask(rle, orig_w, orig_h)
                except Exception:
                    continue

                area_px = int(mask.sum())
                total_px = orig_w * orig_h
                per_class_areas[cls].append(area_px)
                per_class_coverage[cls].append(area_px / total_px if total_px > 0 else 0.0)
                task_classes.add(cls)
                instance_count += 1

            if task_classes:
                for cls in task_classes:
                    per_class_image_count[cls] += 1
                instances_per_task.append(instance_count)
                classes_per_task.append(task_classes)

        progress.update(task_progress, description="Done.")

    n_annotated = len(annotated_tasks)
    n_unannotated = unannotated_count

    # co-occurrence: images with all classes vs partial
    all_classes_set = set(classes)
    n_all_classes = sum(1 for s in classes_per_task if s == all_classes_set)
    n_only_one = sum(1 for s in classes_per_task if len(s) == 1)
    n_both_or_more = sum(1 for s in classes_per_task if len(s) > 1)

    avg_instances = float(np.mean(instances_per_task)) if instances_per_task else 0.0
    median_instances = float(np.median(instances_per_task)) if instances_per_task else 0.0

    # --- print overview ---
    console.print("\n[bold]Dataset Overview[/bold]")
    overview_table = Table(show_header=True, header_style="bold")
    overview_table.add_column("Metric")
    overview_table.add_column("Value", justify="right")
    overview_table.add_row("Total tasks", str(total_tasks))
    overview_table.add_row("Annotated tasks", f"{n_annotated} ({n_annotated / total_tasks:.1%})" if total_tasks else "0")
    overview_table.add_row("Unannotated tasks", str(n_unannotated))
    overview_table.add_row("Avg brush instances / task", f"{avg_instances:.1f}")
    overview_table.add_row("Median brush instances / task", f"{median_instances:.0f}")
    if len(classes) > 1:
        overview_table.add_row(f"Tasks with all {len(classes)} classes", str(n_all_classes))
        overview_table.add_row("Tasks with only 1 class", str(n_only_one))
    console.print(overview_table)

    # --- print per-class stats ---
    console.print("\n[bold]Per-Class Statistics[/bold]")
    cls_table = Table(show_header=True, header_style="bold")
    cls_table.add_column("Class")
    cls_table.add_column("Images", justify="right")
    cls_table.add_column("Images %", justify="right")
    cls_table.add_column("Instances", justify="right")
    cls_table.add_column("Mean area (px)", justify="right")
    cls_table.add_column("Median area (px)", justify="right")
    cls_table.add_column("Std area (px)", justify="right")
    cls_table.add_column("Mean coverage %", justify="right")

    for cls in classes:
        areas = per_class_areas[cls]
        coverages = per_class_coverage[cls]
        img_count = per_class_image_count[cls]
        img_pct = img_count / n_annotated if n_annotated else 0.0
        if areas:
            mean_area = float(np.mean(areas))
            median_area = float(np.median(areas))
            std_area = float(np.std(areas))
            mean_cov = float(np.mean(coverages)) * 100
        else:
            mean_area = median_area = std_area = mean_cov = 0.0
        cls_table.add_row(
            cls,
            str(img_count),
            f"{img_pct:.1%}",
            str(len(areas)),
            f"{mean_area:,.0f}",
            f"{median_area:,.0f}",
            f"{std_area:,.0f}",
            f"{mean_cov:.2f}%",
        )

    console.print(cls_table)

    if len(classes) > 1 and n_annotated:
        console.print("\n[bold]Class Co-occurrence[/bold]")
        cooc_table = Table(show_header=True, header_style="bold")
        cooc_table.add_column("Combination")
        cooc_table.add_column("Images", justify="right")
        cooc_table.add_column("Images %", justify="right")

        from itertools import combinations

        # single-class rows
        for cls in classes:
            only_this = sum(1 for s in classes_per_task if s == {cls})
            cooc_table.add_row(
                f"Only {cls}",
                str(only_this),
                f"{only_this / n_annotated:.1%}",
            )
        # multi-class rows (pairs, triples, …)
        for r in range(2, len(classes) + 1):
            for combo in combinations(classes, r):
                combo_set = set(combo)
                count = sum(1 for s in classes_per_task if s == combo_set)
                label = " + ".join(combo)
                cooc_table.add_row(label, str(count), f"{count / n_annotated:.1%}")

        console.print(cooc_table)

    # --- optional JSON export ---
    if export:
        export_path = export if export.suffix else export.with_suffix(".json")
        result = {
            "project_id": pid,
            "project_title": project_title,
            "classes": classes,
            "overview": {
                "total_tasks": total_tasks,
                "annotated_tasks": n_annotated,
                "unannotated_tasks": n_unannotated,
                "avg_instances_per_task": round(avg_instances, 2),
                "median_instances_per_task": round(median_instances, 2),
            },
            "per_class": {
                cls: {
                    "image_count": per_class_image_count[cls],
                    "image_pct": round(per_class_image_count[cls] / n_annotated, 4) if n_annotated else 0,
                    "instance_count": len(per_class_areas[cls]),
                    "mean_area_px": round(float(np.mean(per_class_areas[cls])), 1) if per_class_areas[cls] else 0,
                    "median_area_px": round(float(np.median(per_class_areas[cls])), 1) if per_class_areas[cls] else 0,
                    "std_area_px": round(float(np.std(per_class_areas[cls])), 1) if per_class_areas[cls] else 0,
                    "mean_coverage_pct": round(float(np.mean(per_class_coverage[cls])) * 100, 4) if per_class_coverage[cls] else 0,
                }
                for cls in classes
            },
        }
        if len(classes) > 1:
            from itertools import combinations
            cooc: dict[str, int] = {}
            for cls in classes:
                cooc[cls] = sum(1 for s in classes_per_task if s == {cls})
            for r in range(2, len(classes) + 1):
                for combo in combinations(classes, r):
                    key = " + ".join(combo)
                    cooc[key] = sum(1 for s in classes_per_task if s == set(combo))
            result["co_occurrence"] = cooc

        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(result, indent=2))
        success(f"\nSaved → {export_path.resolve()}")
