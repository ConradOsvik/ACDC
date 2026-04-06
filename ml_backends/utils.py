import os
import tempfile
import uuid
from urllib.parse import quote
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
