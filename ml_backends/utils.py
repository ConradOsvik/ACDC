"""Shared helpers for the YOLO and SAM3 ML backends.

Auth: we delegate to the official label-studio-sdk client, which transparently
handles both legacy "Token" auth and JWT refresh-token auth depending on the
shape of LABEL_STUDIO_API_KEY. Anything not exposed by the SDK (binary image
downloads via /tasks/<id>/resolve) goes through requests with the SDK's
auth headers.
"""

import base64
import json
import os
import random
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw
from label_studio_converter.brush import mask2rle
from label_studio_sdk.client import LabelStudio

LS_URL = os.environ.get('LABEL_STUDIO_URL', 'http://localhost:8080').rstrip('/')
LS_API_KEY = os.environ.get('LABEL_STUDIO_API_KEY', '')

_client: LabelStudio | None = None


def get_ls_client() -> LabelStudio:
    global _client
    if _client is None:
        _client = LabelStudio(base_url=LS_URL, api_key=LS_API_KEY)
    return _client


def _auth_headers() -> dict:
    return get_ls_client()._client_wrapper.get_headers()


def ls_get(url: str, **kwargs) -> requests.Response:
    """GET against the LS server. Accepts a path or a full URL."""
    if not url.startswith('http'):
        url = f'{LS_URL}{url}' if url.startswith('/') else f'{LS_URL}/{url}'
    resp = requests.get(url, headers=_auth_headers(), **kwargs)
    resp.raise_for_status()
    return resp


def get_image_path(image_uri: str, task_id: int) -> str:
    """Download the image referenced by an LS task and return a local path."""
    if image_uri.startswith(('s3://', 'gs://', 'azure://')):
        fileuri_b64 = base64.b64encode(image_uri.encode()).decode()
        url = f'/tasks/{task_id}/resolve/?fileuri={fileuri_b64}'
        resp = ls_get(url, timeout=30, allow_redirects=True)
    elif image_uri.startswith('/data/'):
        resp = ls_get(image_uri, timeout=30)
    else:
        resp = requests.get(image_uri, timeout=30)
        resp.raise_for_status()

    suffix = os.path.splitext(image_uri.split('?')[0])[-1] or '.jpg'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


def get_brush_label_config(parsed_label_config: dict, tag: str) -> tuple[str, str, list[str]]:
    """Extract from_name, to_name, and labels from a parsed Label Studio config."""
    from_name, to_name, labels = 'brush_label', 'image', []
    for tag_name, tag_info in parsed_label_config.items():
        if tag_info.get('type', '').lower() == tag.lower():
            from_name = tag_name
            to_name = tag_info.get('to_name', ['image'])[0]
            labels = tag_info.get('labels', [])
            break
    return from_name, to_name, labels


def polygons_to_mask(polygons_xy, height: int, width: int) -> np.ndarray:
    """Draw multiple polygon contours onto a single binary mask (union)."""
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for polygon_xy in polygons_xy:
        if len(polygon_xy) < 3:
            continue
        draw.polygon([(float(x), float(y)) for x, y in polygon_xy], fill=255)
    return np.array(mask_img)


def rle_to_mask(rle: list, width: int, height: int) -> np.ndarray:
    from label_studio_sdk.converter.brush import decode_rle
    decoded = np.array(decode_rle(rle), dtype=np.uint8).reshape(height, width, 4)
    return (decoded[:, :, 3] > 0).astype(np.uint8)


def mask_to_yolo_polygons(mask: np.ndarray, min_area_px: int = 25) -> list[list[float]]:
    """Return one normalised polygon per disconnected region in the mask."""
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


def export_annotations_to_yolo(
    project_id: int, output_dir: Path, split: float = 0.8, seed: int | None = None,
    test_split: float = 0.0, max_images: int | None = None,
) -> tuple[list[str], int, int]:
    """Fetch annotated tasks from LS and write a YOLO segmentation dataset.

    Returns (class_names, exported_count, skipped_count).
    """
    project = ls_get(f'/api/projects/{project_id}/', timeout=10).json()
    xml_root = ET.fromstring(project['label_config'])
    classes: list[str] = []
    for brush_labels in xml_root.findall('.//BrushLabels'):
        for label_el in brush_labels.findall('Label'):
            v = label_el.get('value')
            if v and v not in classes:
                classes.append(v)

    if not classes:
        raise ValueError('No BrushLabels found in project label config')

    if max_images is not None:
        # Fetch only annotated task IDs (lightweight), randomly sample, then export just those.
        all_ids: list[int] = []
        page_size = 1000
        page = 1
        while True:
            resp = ls_get(
                f'/api/tasks?project={project_id}&page_size={page_size}&page={page}',
                timeout=60,
            ).json()
            tasks_page = resp.get('tasks', [])
            # keep only tasks that have at least one annotation
            all_ids.extend(t['id'] for t in tasks_page if t.get('total_annotations', 0) > 0)
            total = resp.get('total', 0)
            if len(all_ids) >= total or not tasks_page or len(tasks_page) < page_size:
                break
            page += 1
        random.seed(seed)
        random.shuffle(all_ids)
        selected_ids = all_ids[:max_images]
        ids_qs = '&'.join(f'ids[]={i}' for i in selected_ids)
        tasks = ls_get(
            f'/api/projects/{project_id}/export?exportType=JSON&{ids_qs}',
            timeout=120,
        ).json()
        print(f'[export] Sampled {len(tasks)} of {len(all_ids)} annotated tasks', flush=True)
    else:
        tasks = ls_get(f'/api/projects/{project_id}/export?exportType=JSON', timeout=120).json()

    if len(tasks) < 2:
        raise ValueError(
            f"Need at least 2 tasks to build a train/val split (project has {len(tasks)}). "
            "Upload more images and annotate them, then click Start Training again."
        )

    random.seed(seed)
    random.shuffle(tasks)
    n = len(tasks)

    n_test = max(1, int(n * test_split)) if test_split > 0 else 0
    test_tasks = tasks[:n_test]
    remaining = tasks[n_test:]

    n_rem = len(remaining)
    split_idx = max(1, min(n_rem - 1, int(n_rem * split)))
    splits: dict[str, list] = {'train': remaining[:split_idx], 'val': remaining[split_idx:]}
    if test_tasks:
        splits['test'] = test_tasks

    for split_name in splits:
        (output_dir / 'images' / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split_name).mkdir(parents=True, exist_ok=True)

    yaml_lines = [
        f'path: {output_dir.resolve()}',
        'train: images/train',
        'val: images/val',
    ]
    if test_tasks:
        yaml_lines.append('test: images/test')
    yaml_lines += [f'nc: {len(classes)}', f'names: {classes}']
    (output_dir / 'data.yaml').write_text('\n'.join(yaml_lines) + '\n')

    if test_tasks:
        (output_dir / 'test_ids.json').write_text(
            json.dumps([t['id'] for t in test_tasks])
        )

    exported = skipped = 0

    for split_name, split_tasks in splits.items():
        for task in split_tasks:
            task_id = task['id']
            annotations = [a for a in task.get('annotations', []) if not a.get('was_cancelled')]
            if not annotations:
                skipped += 1
                continue

            brush_results = [
                r for a in annotations for r in a.get('result', [])
                if r.get('type') == 'brushlabels'
            ]
            is_negative = not brush_results

            stem = f'task_{task_id:06d}'
            yolo_lines: list[str] = []

            for r in brush_results:
                value = r.get('value', {})
                rle = value.get('rle', [])
                label_names = value.get('brushlabels', [])
                if not rle or not label_names or label_names[0] not in classes:
                    continue
                class_idx = classes.index(label_names[0])
                orig_w = r.get('original_width', 100)
                orig_h = r.get('original_height', 100)
                try:
                    mask = rle_to_mask(rle, orig_w, orig_h)
                    polygons = mask_to_yolo_polygons(mask)
                except Exception as e:
                    print(f'  [WARN] task {task_id}: decode error: {e}')
                    continue
                for polygon in polygons:
                    coords = ' '.join(f'{v:.6f}' for v in polygon)
                    yolo_lines.append(f'{class_idx} {coords}')

            if not is_negative and not yolo_lines:
                skipped += 1
                continue

            (output_dir / 'labels' / split_name / f'{stem}.txt').write_text('\n'.join(yolo_lines))

            image_uri = task.get('data', {}).get('image', '')
            ext = Path(image_uri.split('?')[0]).suffix or '.jpg'
            img_dest = output_dir / 'images' / split_name / f'{stem}{ext}'
            try:
                if image_uri.startswith(('s3://', 'gs://', 'azure://')):
                    fileuri_b64 = base64.b64encode(image_uri.encode()).decode()
                    resp = ls_get(
                        f'/tasks/{task_id}/resolve/?fileuri={fileuri_b64}',
                        timeout=30, allow_redirects=True,
                    )
                elif image_uri.startswith('/data/'):
                    resp = ls_get(image_uri, timeout=30)
                else:
                    resp = requests.get(image_uri, timeout=30)
                    resp.raise_for_status()
                img_dest.write_bytes(resp.content)
            except Exception as e:
                print(f'  [WARN] Could not download image for task {task_id}: {e}')

            exported += 1

    return classes, exported, skipped


def polygons_to_brush_result(
    polygons_xy, height: int, width: int, label_name: str, from_name: str, to_name: str
) -> dict:
    """Merge all polygons for a label into a single filled brush (RLE) region."""
    mask_np = polygons_to_mask(polygons_xy, height, width)
    rle = mask2rle(mask_np)
    return {
        "id": uuid.uuid4().hex[:8],
        "from_name": from_name,
        "to_name": to_name,
        "type": "brushlabels",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "value": {
            "format": "rle",
            "rle": rle,
            "brushlabels": [label_name],
        },
    }
