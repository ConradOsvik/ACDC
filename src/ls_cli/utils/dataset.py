"""Dataset export & visualization helpers (Label Studio → YOLO segmentation)."""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from ls_cli.client import auth_headers
from ls_cli.config import Settings

if TYPE_CHECKING:
    from label_studio_sdk.client import LabelStudio


def _rle_to_mask(rle: list, width: int, height: int):
    import numpy as np
    from label_studio_sdk.converter.brush import decode_rle

    decoded = np.array(decode_rle(rle), dtype=np.uint8).reshape(height, width, 4)
    return (decoded[:, :, 3] > 0).astype(np.uint8)


def _mask_to_polygons(mask, min_area_px: int = 25) -> list[list[float]]:
    import cv2

    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.squeeze()
        if points.ndim < 2 or len(points) < 3:
            continue
        polygons.append([coord for x, y in points for coord in (float(x) / w, float(y) / h)])
    return polygons


def _classes_from_label_config(label_config_xml: str) -> list[str]:
    classes: list[str] = []
    for brush_labels in ET.fromstring(label_config_xml).findall(".//BrushLabels"):
        for label_el in brush_labels.findall("Label"):
            v = label_el.get("value")
            if v and v not in classes:
                classes.append(v)
    return classes


def export_yolo_dataset(
    settings: Settings,
    ls: LabelStudio,
    project_id: int,
    output_dir: Path,
    *,
    split: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], int, int]:
    """Export annotated tasks from a project as a YOLO segmentation dataset.

    Returns (classes, exported_count, skipped_count).
    """
    import requests

    project = ls.projects.get(id=project_id)
    classes = _classes_from_label_config(project.label_config)
    if not classes:
        raise SystemExit("No BrushLabels found in project label config")

    base = settings.label_studio_url.rstrip("/")
    headers = auth_headers(ls)

    resp = requests.get(
        f"{base}/api/projects/{project_id}/export?exportType=JSON",
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    tasks = resp.json()

    if len(tasks) < 2:
        raise SystemExit(
            f"Need at least 2 tasks to build a train/val split (project has {len(tasks)}). "
            "Upload more images and annotate them first."
        )

    random.seed(seed)
    random.shuffle(tasks)
    n = len(tasks)
    split_idx = max(1, min(n - 1, int(n * split)))
    splits = {"train": tasks[:split_idx], "val": tasks[split_idx:]}

    for split_name in splits:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    (output_dir / "data.yaml").write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )

    exported = skipped = 0

    for split_name, split_tasks in splits.items():
        for task in split_tasks:
            task_id = task["id"]
            annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
            brush_results = [
                r for a in annotations for r in a.get("result", [])
                if r.get("type") == "brushlabels"
            ]
            if not brush_results:
                skipped += 1
                continue

            stem = f"task_{task_id:06d}"
            yolo_lines: list[str] = []

            for r in brush_results:
                value = r.get("value", {})
                rle = value.get("rle", [])
                label_names = value.get("brushlabels", [])
                if not rle or not label_names or label_names[0] not in classes:
                    continue
                class_idx = classes.index(label_names[0])
                orig_w = r.get("original_width", 100)
                orig_h = r.get("original_height", 100)
                try:
                    mask = _rle_to_mask(rle, orig_w, orig_h)
                    polygons = _mask_to_polygons(mask)
                except Exception as e:
                    print(f"  [WARN] task {task_id}: decode error: {e}")
                    continue
                for polygon in polygons:
                    coords = " ".join(f"{v:.6f}" for v in polygon)
                    yolo_lines.append(f"{class_idx} {coords}")

            if not yolo_lines:
                skipped += 1
                continue

            (output_dir / "labels" / split_name / f"{stem}.txt").write_text("\n".join(yolo_lines))

            image_uri = task.get("data", {}).get("image", "")
            ext = Path(image_uri.split("?")[0]).suffix or ".jpg"
            img_dest = output_dir / "images" / split_name / f"{stem}{ext}"
            try:
                if image_uri.startswith(("s3://", "gs://", "azure://")):
                    import base64
                    fileuri_b64 = base64.b64encode(image_uri.encode()).decode()
                    img_resp = requests.get(
                        f"{base}/tasks/{task_id}/resolve/?fileuri={fileuri_b64}",
                        headers=headers, timeout=30, allow_redirects=True,
                    )
                    img_resp.raise_for_status()
                elif image_uri.startswith("/data/"):
                    img_resp = requests.get(f"{base}{image_uri}", headers=headers, timeout=30)
                    img_resp.raise_for_status()
                else:
                    img_resp = requests.get(image_uri, timeout=30)
                    img_resp.raise_for_status()
                img_dest.write_bytes(img_resp.content)
            except Exception as e:
                print(f"  [WARN] Could not download image for task {task_id}: {e}")

            exported += 1

    return classes, exported, skipped


def visualize_yolo_dataset(
    dataset_dir: Path,
    *,
    split: str = "all",
    max_images: int | None = None,
    save_dir: Path | None = None,
) -> tuple[int, list[str]]:
    """Render image+polygon overlays from a YOLO seg dataset.

    Returns (count_shown, issues).
    """
    import cv2
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import yaml
    from matplotlib.colors import hsv_to_rgb

    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"data.yaml not found in {dataset_dir}. Run export first.")

    data_cfg = yaml.safe_load(yaml_path.read_text())
    class_names: list[str] = data_cfg.get("names", [])
    nc = len(class_names)

    def class_color_rgb(idx: int) -> tuple[float, float, float]:
        hue = (idx / max(nc, 1)) % 1.0
        return tuple(hsv_to_rgb([hue, 0.85, 0.95]))

    splits = ["train", "val"] if split == "all" else [split]

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    total_shown = 0
    issues: list[str] = []

    for split_name in splits:
        img_dir = dataset_dir / "images" / split_name
        lbl_dir = dataset_dir / "labels" / split_name
        if not img_dir.exists():
            continue

        img_paths = sorted(img_dir.glob("*"))
        if max_images is not None:
            img_paths = img_paths[: max_images - total_shown]

        for img_path in img_paths:
            if max_images is not None and total_shown >= max_images:
                break

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                issues.append(f"{split_name}/{img_path.name}: could not read image")
                continue

            h, w = img_bgr.shape[:2]
            polygons: list[tuple[int, list[tuple[float, float]]]] = []
            if lbl_path.exists():
                for line in lbl_path.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) < 7:
                        issues.append(f"{split_name}/{img_path.name}: degenerate polygon ({line!r})")
                        continue
                    cls_idx = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if len(coords) % 2 != 0:
                        issues.append(f"{split_name}/{img_path.name}: odd coord count for class {cls_idx}")
                        continue
                    pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
                    polygons.append((cls_idx, pts))
            else:
                issues.append(f"{split_name}/{img_path.name}: missing label file")

            fig, ax = plt.subplots(figsize=(10, 10 * h / w))
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{split_name}/{img_path.name}  ({w}×{h})  —  {len(polygons)} annotation(s)")
            ax.axis("off")

            legend_handles: dict[int, mpatches.Patch] = {}
            for cls_idx, pts in polygons:
                color = class_color_rgb(cls_idx)
                poly = mpatches.Polygon(
                    pts, closed=True, fill=True,
                    facecolor=(*color, 0.25), edgecolor=color, linewidth=1.5,
                )
                ax.add_patch(poly)
                if cls_idx not in legend_handles:
                    label = class_names[cls_idx] if cls_idx < nc else f"class_{cls_idx}"
                    legend_handles[cls_idx] = mpatches.Patch(color=color, label=label)

            if legend_handles:
                ax.legend(handles=list(legend_handles.values()), loc="upper right",
                          fontsize=8, framealpha=0.7)

            plt.tight_layout(pad=0.5)
            if save_dir:
                out_path = save_dir / f"{split_name}_{img_path.stem}.png"
                plt.savefig(out_path, dpi=120)
                plt.close(fig)
                print(f"  Saved {out_path}")
            else:
                plt.show()
                plt.close(fig)

            total_shown += 1

    return total_shown, issues
