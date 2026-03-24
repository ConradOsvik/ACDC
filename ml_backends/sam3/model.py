import os
import uuid
import tempfile
import requests
import numpy as np
from PIL import Image, ImageDraw
from label_studio_ml.model import LabelStudioMLBase
from label_studio_converter.brush import mask2rle
from ultralytics.models.sam import SAM3SemanticPredictor

LS_URL = os.environ.get('LABEL_STUDIO_URL', 'http://localhost:8080')
LS_API_KEY = os.environ.get('LABEL_STUDIO_API_KEY', '')
MODEL_PATH = os.environ.get('SAM3_MODEL_PATH', 'sam3.pt')

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


def _fetch(url: str, **kwargs) -> requests.Response:
    token = _get_access_token()
    resp = requests.get(url, headers={'Authorization': f'Bearer {token}'}, **kwargs)
    if resp.status_code == 401:
        token = _get_access_token(force_refresh=True)
        resp = requests.get(url, headers={'Authorization': f'Bearer {token}'}, **kwargs)
    resp.raise_for_status()
    return resp


def get_image_path(image_uri: str, task_id: int) -> str:
    if image_uri.startswith('s3://') or image_uri.startswith('/data/'):
        presign_url = f'{LS_URL}/tasks/{task_id}/presign/?fileuri={requests.utils.quote(image_uri, safe="")}'
        resp = _fetch(presign_url, timeout=30, allow_redirects=True)
    else:
        resp = _fetch(image_uri, timeout=30)

    suffix = os.path.splitext(image_uri.split('?')[0])[-1] or '.jpg'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name



def polygons_to_mask(polygons_xy, height, width) -> np.ndarray:
    """Draw multiple polygon contours onto a single binary mask (union)."""
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for polygon_xy in polygons_xy:
        if len(polygon_xy) < 3:
            continue
        draw.polygon([(float(x), float(y)) for x, y in polygon_xy], fill=255)
    return np.array(mask_img)  # 0 or 255, uint8 — mask2rle thresholds at 128


class SAM3Backend(LabelStudioMLBase):
    def __init__(self, project_id=None, **kwargs):
        super().__init__(**kwargs)
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model=MODEL_PATH,
            save=False,
            verbose=False,
        )
        print(f"Loading SAM 3 model from {MODEL_PATH}...")
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def _get_label_config(self):
        from_name, to_name, labels = 'brush_label', 'image', []
        if self.parsed_label_config:
            for tag_name, tag_info in self.parsed_label_config.items():
                if tag_info.get('type', '').lower() == 'brushlabels':
                    from_name = tag_name
                    to_name = tag_info.get('to_name', ['image'])[0]
                    labels = tag_info.get('labels', [])
                    break
        print(f"[SAM3] from_name={from_name}, labels={labels}")
        return from_name, to_name, labels

    def predict(self, tasks, context=None, **kwargs):
        from_name, to_name, labels = self._get_label_config()
        predictions = []

        for task in tasks:
            local_path = get_image_path(task['data']['image'], task_id=task['id'])
            print(f"[SAM3] Loading: {local_path}")
            image = Image.open(local_path).convert("RGB")
            width, height = image.size

            try:
                self.predictor.set_image(local_path)
                result_list = []

                # Interactive bbox mode
                input_box = None
                if context and context.get('result'):
                    for item in context['result']:
                        if item['type'] == 'rectanglelabels':
                            v = item['value']
                            x1 = v['x'] * width / 100.0
                            y1 = v['y'] * height / 100.0
                            x2 = (v['x'] + v['width']) * width / 100.0
                            y2 = (v['y'] + v['height']) * height / 100.0
                            input_box = [x1, y1, x2, y2]
                            break

                if input_box:
                    label_name = labels[0] if labels else "Object"
                    print(f"[SAM3] Interactive box mode, label: {label_name}")
                    results = self.predictor(bboxes=[input_box])
                    if results and results[0].masks is not None:
                        polygons = results[0].masks.xy
                        print(f"[SAM3] {len(polygons)} masks → merging into 1 brush region")
                        result_list.append(
                            self._polygons_to_brush_result(polygons, height, width, label_name, from_name, to_name)
                        )
                else:
                    query_labels = labels if labels else ["object"]
                    print(f"[SAM3] Text mode, querying: {query_labels}")
                    for label_name in query_labels:
                        results = self.predictor(text=[label_name.lower()])
                        if results and results[0].masks is not None:
                            polygons = results[0].masks.xy
                            print(f"[SAM3] '{label_name}': {len(polygons)} masks → 1 brush region")
                            result_list.append(
                                self._polygons_to_brush_result(polygons, height, width, label_name, from_name, to_name)
                            )
                        else:
                            print(f"[SAM3] '{label_name}': no masks")
            finally:
                pass

            print(f"[SAM3] Returning {len(result_list)} brush regions")
            predictions.append({"result": result_list})

        return predictions

    def _polygons_to_brush_result(self, polygons_xy, height, width, label_name, from_name, to_name):
        """Merge all polygons for a label into a single filled brush (RLE) region."""
        mask_np = polygons_to_mask(polygons_xy, height, width)  # 0 or 255, uint8
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
            }
        }
