"""
CLI for exporting Label Studio annotations and triggering YOLO training.

Training runs inside the YOLO container — this script just tells Label Studio
to start it. The container must be running.

Commands:
    uv run scripts/yolo.py export
    uv run scripts/yolo.py export --output-dir ./dataset --split 0.8 --project-id 1

    uv run scripts/yolo.py train
    uv run scripts/yolo.py train --project-id 1

    uv run scripts/yolo.py visualize
    uv run scripts/yolo.py visualize --dataset-dir ./exports/yolo_dataset --split all --max-images 20
    uv run scripts/yolo.py visualize --dataset-dir ./exports/yolo_dataset --save-dir ./exports/viz
"""

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote
import os

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

LS_URL = os.environ["LABEL_STUDIO_URL"].rstrip("/")
REFRESH_TOKEN = os.environ["LABEL_STUDIO_API_KEY"]

_access_token: str = ""


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


def ls_post(path: str, **kwargs) -> requests.Response:
    token = _get_access_token()
    resp = requests.post(f"{LS_URL}{path}", headers={"Authorization": f"Bearer {token}"}, **kwargs)
    if resp.status_code == 401:
        token = _get_access_token(force_refresh=True)
        resp = requests.post(f"{LS_URL}{path}", headers={"Authorization": f"Bearer {token}"}, **kwargs)
    resp.raise_for_status()
    return resp


def rle_to_mask(rle: list, width: int, height: int) -> np.ndarray:
    from label_studio_sdk.converter.brush import decode_rle
    decoded = np.array(decode_rle(rle), dtype=np.uint8).reshape(height, width, 4)
    return (decoded[:, :, 3] > 0).astype(np.uint8)


def mask_to_yolo_polygons(mask: np.ndarray, min_area_px: int = 25) -> list[list[float]]:
    """Return one normalised polygon per disconnected region in the mask.

    Uses an absolute pixel minimum (default 25 px) to drop single-pixel RLE
    artefacts while keeping even very small real annotations.
    """
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


def resolve_project(project_id: int | None) -> dict:
    projects = ls_get("/api/projects/").json()["results"]
    if project_id:
        project = next((p for p in projects if p["id"] == project_id), None)
        if not project:
            raise SystemExit(f"Project id {project_id} not found")
    elif len(projects) == 1:
        project = projects[0]
    else:
        print("Multiple projects — specify --project-id:")
        for p in projects:
            print(f"  {p['id']}: {p['title']}")
        raise SystemExit(1)
    return project


def cmd_export(args: argparse.Namespace) -> None:
    project = resolve_project(args.project_id)
    print(f"Project: {project['title']} (id={project['id']})")

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

    print("Fetching annotations from Label Studio...")
    tasks = ls_get(f"/api/projects/{project['id']}/export?exportType=JSON").json()
    print(f"  {len(tasks)} tasks")

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
                    mask = rle_to_mask(rle, orig_w, orig_h)
                    polygons = mask_to_yolo_polygons(mask)
                except Exception as e:
                    print(f"  [WARN] task {task_id}: decode error: {e}")
                    continue
                for polygon in polygons:
                    coords = " ".join(f"{v:.6f}" for v in polygon)
                    yolo_lines.append(f"{class_idx} {coords}")

            if not yolo_lines:
                skipped += 1
                continue

            (out / "labels" / split_name / f"{stem}.txt").write_text("\n".join(yolo_lines))

            image_uri = task.get("data", {}).get("image", "")
            ext = Path(image_uri.split("?")[0]).suffix or ".jpg"
            img_dest = out / "images" / split_name / f"{stem}{ext}"
            try:
                if image_uri.startswith("/data/") or image_uri.startswith("s3://"):
                    resp = ls_get(
                        f"/tasks/{task_id}/presign/?fileuri={quote(image_uri, safe='')}",
                        timeout=30, allow_redirects=True,
                    )
                else:
                    resp = requests.get(image_uri, timeout=30)
                    resp.raise_for_status()
                img_dest.write_bytes(resp.content)
            except Exception as e:
                print(f"  [WARN] Could not download image for task {task_id}: {e}")

            exported += 1

    print(f"\nDone: {exported} exported, {skipped} skipped (no brush annotations)")
    print(f"Dataset written to: {out.resolve()}")


def cmd_visualize(args: argparse.Namespace) -> None:
    import cv2
    import yaml
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import hsv_to_rgb

    dataset_dir = args.dataset_dir
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"data.yaml not found in {dataset_dir}. Run 'export' first.")

    with open(yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    class_names: list[str] = data_cfg.get("names", [])
    nc = len(class_names)

    # Generate a distinct color per class (HSV wheel → BGR for cv2, RGB for matplotlib)
    def class_color_rgb(idx: int) -> tuple[float, float, float]:
        hue = (idx / max(nc, 1)) % 1.0
        return tuple(hsv_to_rgb([hue, 0.85, 0.95]))

    splits = ["train", "val"] if args.split == "all" else [args.split]

    save_dir: Path | None = args.save_dir
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    total_shown = 0
    issues: list[str] = []

    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.exists():
            print(f"  [SKIP] {img_dir} does not exist")
            continue

        img_paths = sorted(img_dir.glob("*"))
        if args.max_images:
            img_paths = img_paths[: args.max_images - total_shown]

        for img_path in img_paths:
            if total_shown >= (args.max_images or 10**9):
                break

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                issues.append(f"{split}/{img_path.name}: could not read image")
                continue

            h, w = img_bgr.shape[:2]

            if not lbl_path.exists():
                issues.append(f"{split}/{img_path.name}: missing label file")
                # Still show the image so the gap is visible
                polygons = []
            else:
                polygons = []
                for line in lbl_path.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) < 7:  # class + at least 3 points
                        issues.append(f"{split}/{img_path.name}: degenerate polygon ({line!r})")
                        continue
                    cls_idx = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if len(coords) % 2 != 0:
                        issues.append(f"{split}/{img_path.name}: odd coord count for class {cls_idx}")
                        continue
                    pts = [(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
                    polygons.append((cls_idx, pts))

            # Draw with matplotlib for clean display
            fig, ax = plt.subplots(figsize=(10, 10 * h / w))
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{split}/{img_path.name}  ({w}×{h})  —  {len(polygons)} annotation(s)")
            ax.axis("off")

            legend_handles: dict[int, mpatches.Patch] = {}
            for cls_idx, pts in polygons:
                color = class_color_rgb(cls_idx)
                poly = mpatches.Polygon(pts, closed=True, fill=True,
                                   facecolor=(*color, 0.25), edgecolor=color, linewidth=1.5)
                ax.add_patch(poly)
                if cls_idx not in legend_handles:
                    label = class_names[cls_idx] if cls_idx < nc else f"class_{cls_idx}"
                    legend_handles[cls_idx] = mpatches.Patch(color=color, label=label)

            if legend_handles:
                ax.legend(handles=list(legend_handles.values()), loc="upper right",
                          fontsize=8, framealpha=0.7)

            plt.tight_layout(pad=0.5)
            if save_dir:
                out_path = save_dir / f"{split}_{img_path.stem}.png"
                plt.savefig(out_path, dpi=120)
                plt.close(fig)
                print(f"  Saved {out_path}")
            else:
                plt.show()
                plt.close(fig)

            total_shown += 1

    print(f"\nVisualized {total_shown} image(s).")
    if issues:
        print(f"\n{len(issues)} issue(s) found:")
        for msg in issues:
            print(f"  [!] {msg}")
    else:
        print("No issues detected.")


def cmd_train() -> None:
    backend_url = os.environ.get("YOLO_BACKEND_URL", "http://localhost:9090").rstrip("/")

    print(f"Triggering training on {backend_url}...")
    resp = requests.post(f"{backend_url}/train", json={}, timeout=None)
    resp.raise_for_status()
    print("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="yolo")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Export annotated tasks as a YOLO segmentation dataset")
    exp.add_argument("--project-id", type=int, default=None)
    exp.add_argument("--output-dir", type=Path, default=Path("exports/yolo_dataset"))
    exp.add_argument("--split", type=float, default=0.8, help="Train fraction (rest goes to val)")
    exp.add_argument("--seed", type=int, default=42)

    sub.add_parser("train", help="Trigger YOLO training inside the running container")

    viz = sub.add_parser("visualize", help="Visualize exported YOLO dataset with polygon overlays")
    viz.add_argument("--dataset-dir", type=Path, default=Path("exports/yolo_dataset"))
    viz.add_argument("--split", choices=["train", "val", "all"], default="all")
    viz.add_argument("--max-images", type=int, default=None, metavar="N",
                     help="Stop after N images (default: all)")
    viz.add_argument("--save-dir", type=Path, default=None, metavar="DIR",
                     help="Save images to DIR instead of displaying interactively")

    args = parser.parse_args()
    if args.command == "export":
        cmd_export(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    else:
        cmd_train()


if __name__ == "__main__":
    main()
