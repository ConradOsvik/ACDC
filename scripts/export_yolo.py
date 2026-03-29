"""
Export Label Studio BrushLabel annotations to YOLO segmentation format.

The standard Label Studio YOLO exporter only handles polygon/rectangle annotations.
This script decodes RLE brush masks back to polygon contours for YOLO training.

Usage:
    uv run scripts/export_yolo.py
    uv run scripts/export_yolo.py --output-dir ./dataset --split 0.8 --project-id 1
"""

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote
import os

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from label_studio_sdk.converter.brush import decode_rle

load_dotenv(Path(__file__).parent / ".env")

LS_URL = os.environ["LABEL_STUDIO_URL"].rstrip("/")
REFRESH_TOKEN = os.environ["LABEL_STUDIO_API_KEY"]

_access_token: str | None = None


def _get_access_token(force_refresh: bool = False) -> str:
    global _access_token
    if _access_token and not force_refresh:
        return _access_token
    resp = requests.post(
        f"{LS_URL}/api/token/refresh/",
        json={"refresh": REFRESH_TOKEN},
        timeout=10,
    )
    resp.raise_for_status()
    _access_token = resp.json()["access"]
    return _access_token


def ls_get(path: str, **kwargs) -> requests.Response:
    token = _get_access_token()
    resp = requests.get(f"{LS_URL}{path}", headers={"Authorization": f"Bearer {token}"}, **kwargs)
    if resp.status_code == 401:
        token = _get_access_token(force_refresh=True)
        resp = requests.get(f"{LS_URL}{path}", headers={"Authorization": f"Bearer {token}"}, **kwargs)
    resp.raise_for_status()
    return resp


def rle_to_mask(rle: list, width: int, height: int) -> np.ndarray:
    decoded = np.array(decode_rle(rle), dtype=np.uint8)
    # decode_rle returns flat RGBA array: [R,G,B,A, R,G,B,A, ...]
    decoded = decoded.reshape(height, width, 4)
    return (decoded[:, :, 3] > 0).astype(np.uint8)


def mask_to_yolo_polygon(mask: np.ndarray) -> list[float] | None:
    """Return flat normalized [x1,y1,x2,y2,...] polygon or None if no contour found."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.squeeze()
    if points.ndim < 2 or len(points) < 3:
        return None
    return [coord for x, y in points for coord in (float(x) / w, float(y) / h)]


def download_image(image_uri: str, task_id: int, dest: Path) -> None:
    try:
        if image_uri.startswith("/data/") or image_uri.startswith("s3://"):
            resp = ls_get(
                f"/tasks/{task_id}/presign/?fileuri={quote(image_uri, safe='')}",
                timeout=30,
                allow_redirects=True,
            )
        elif image_uri.startswith(LS_URL):
            resp = ls_get(image_uri[len(LS_URL):], timeout=30)
        else:
            resp = requests.get(image_uri, timeout=30)
            resp.raise_for_status()
        dest.write_bytes(resp.content)
    except Exception as e:
        print(f"  [WARN] Could not download image for task {task_id}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("exports/yolo_dataset"))
    parser.add_argument("--split", type=float, default=0.8, help="Train fraction (rest goes to val)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve project
    projects = ls_get("/api/projects/").json()["results"]
    if args.project_id:
        project = next((p for p in projects if p["id"] == args.project_id), None)
        if not project:
            raise SystemExit(f"Project id {args.project_id} not found")
    elif len(projects) == 1:
        project = projects[0]
    else:
        print("Multiple projects — specify --project-id:")
        for p in projects:
            print(f"  {p['id']}: {p['title']}")
        raise SystemExit(1)

    print(f"Project: {project['title']} (id={project['id']})")

    # Parse class names from BrushLabels in the label config
    xml_root = ET.fromstring(project["label_config"])
    classes: list[str] = []
    for brush_labels in xml_root.findall(".//BrushLabels"):
        for label_el in brush_labels.findall("Label"):
            v = label_el.get("value")
            if v and v not in classes:
                classes.append(v)

    if not classes:
        raise SystemExit("No BrushLabels found in project label config")
    print(f"Classes: {classes}")

    # Fetch all tasks with annotations
    print("Fetching annotations from Label Studio...")
    tasks = ls_get(f"/api/projects/{project['id']}/export?exportType=JSON").json()
    print(f"  {len(tasks)} tasks")

    # Train/val split
    random.seed(args.seed)
    random.shuffle(tasks)
    split_idx = int(len(tasks) * args.split)
    splits = {"train": tasks[:split_idx], "val": tasks[split_idx:]}

    out = args.output_dir
    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
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
            if not annotations:
                skipped += 1
                continue

            brush_results = [
                r
                for a in annotations
                for r in a.get("result", [])
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
                if not rle or not label_names:
                    continue
                label_name = label_names[0]
                if label_name not in classes:
                    continue
                class_idx = classes.index(label_name)

                orig_w = r.get("original_width", 100)
                orig_h = r.get("original_height", 100)

                try:
                    mask = rle_to_mask(rle, orig_w, orig_h)
                    polygon = mask_to_yolo_polygon(mask)
                except Exception as e:
                    print(f"  [WARN] task {task_id}: decode error: {e}")
                    continue

                if polygon is None:
                    continue

                coords = " ".join(f"{v:.6f}" for v in polygon)
                yolo_lines.append(f"{class_idx} {coords}")

            if not yolo_lines:
                skipped += 1
                continue

            (out / "labels" / split_name / f"{stem}.txt").write_text("\n".join(yolo_lines))

            image_uri = task.get("data", {}).get("image", "")
            ext = Path(image_uri.split("?")[0]).suffix or ".jpg"
            download_image(image_uri, task_id, out / "images" / split_name / f"{stem}{ext}")

            exported += 1

    print(f"\nDone: {exported} exported, {skipped} skipped (no brush annotations)")
    print(f"Dataset written to: {out.resolve()}")
    print(f"Train with: yolo train data={out.resolve() / 'data.yaml'} task=segment")


if __name__ == "__main__":
    main()
