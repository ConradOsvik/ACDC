import os
import random
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote
import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw
from label_studio_converter.brush import mask2rle

LS_URL = os.environ.get('LABEL_STUDIO_URL', 'http://localhost:8080')
LS_API_KEY = os.environ.get('LABEL_STUDIO_API_KEY', '')

_access_token_cache = {'token': None}


def _get_access_token(force_refresh: bool = False) -> str:
    if _access_token_cache['token'] and not force_refresh:
        return _access_token_cache['token']
    resp = requests.post(
        f'{LS_URL}/api/token/refresh/',
        json={'refresh': LS_API_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json()['access']
    _access_token_cache['token'] = token
    return token


def ls_get(url: str, **kwargs) -> requests.Response:
    token = _get_access_token()
    resp = requests.get(url, headers={'Authorization': f'Bearer {token}'}, **kwargs)
    if resp.status_code == 401:
        token = _get_access_token(force_refresh=True)
        resp = requests.get(url, headers={'Authorization': f'Bearer {token}'}, **kwargs)
    resp.raise_for_status()
    return resp


def get_image_path(image_uri: str, task_id: int) -> str:
    if image_uri.startswith('s3://') or image_uri.startswith('/data/'):
        presign_url = f'{LS_URL}/tasks/{task_id}/presign/?fileuri={quote(image_uri, safe="")}'
        resp = ls_get(presign_url, timeout=30, allow_redirects=True)
    else:
        resp = ls_get(image_uri, timeout=30)

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
    return np.array(mask_img)  # 0 or 255, uint8 — mask2rle thresholds at 128


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


def export_annotations_to_yolo(project_id: int, output_dir: Path, split: float = 0.8, seed: int = 42) -> tuple[list[str], int, int]:
    """
    Fetch all annotated tasks from Label Studio and write a YOLO segmentation dataset.
    Returns (class_names, exported_count, skipped_count).
    """
    project = ls_get(f'{LS_URL}/api/projects/{project_id}/', timeout=10).json()
    xml_root = ET.fromstring(project['label_config'])
    classes: list[str] = []
    for brush_labels in xml_root.findall('.//BrushLabels'):
        for label_el in brush_labels.findall('Label'):
            v = label_el.get('value')
            if v and v not in classes:
                classes.append(v)

    if not classes:
        raise ValueError('No BrushLabels found in project label config')

    tasks = ls_get(f'{LS_URL}/api/projects/{project_id}/export?exportType=JSON', timeout=120).json()

    random.seed(seed)
    random.shuffle(tasks)
    split_idx = int(len(tasks) * split)
    splits = {'train': tasks[:split_idx], 'val': tasks[split_idx:]}

    for split_name in splits:
        (output_dir / 'images' / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split_name).mkdir(parents=True, exist_ok=True)

    (output_dir / 'data.yaml').write_text(
        f'path: {output_dir.resolve()}\n'
        f'train: images/train\n'
        f'val: images/val\n'
        f'nc: {len(classes)}\n'
        f'names: {classes}\n'
    )

    exported = skipped = 0

    for split_name, split_tasks in splits.items():
        for task in split_tasks:
            task_id = task['id']
            annotations = [a for a in task.get('annotations', []) if not a.get('was_cancelled')]
            brush_results = [
                r for a in annotations for r in a.get('result', [])
                if r.get('type') == 'brushlabels'
            ]
            if not brush_results:
                skipped += 1
                continue

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

            if not yolo_lines:
                skipped += 1
                continue

            (output_dir / 'labels' / split_name / f'{stem}.txt').write_text('\n'.join(yolo_lines))

            image_uri = task.get('data', {}).get('image', '')
            ext = Path(image_uri.split('?')[0]).suffix or '.jpg'
            img_dest = output_dir / 'images' / split_name / f'{stem}{ext}'
            try:
                if image_uri.startswith('/data/') or image_uri.startswith('s3://'):
                    resp = ls_get(
                        f'{LS_URL}/tasks/{task_id}/presign/?fileuri={quote(image_uri, safe="")}',
                        timeout=30, allow_redirects=True,
                    )
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
