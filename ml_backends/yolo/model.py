import io
import os
import uuid
import numpy as np
import requests
from PIL import Image, ImageDraw
from label_studio_ml.model import LabelStudioMLBase
from label_studio_converter.brush import mask2rle
from ultralytics import YOLO

LS_URL = os.environ.get('LABEL_STUDIO_URL', 'http://localhost:8080')
LS_API_KEY = os.environ.get('LABEL_STUDIO_API_KEY', '')
MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', 'yolo11n-seg.pt')
CONF_THRESHOLD = float(os.environ.get('YOLO_CONF', '0.25'))


def get_ls_session():
    session = requests.Session()
    resp = session.post(
        f"{LS_URL}/api/token/refresh/",
        json={"refresh": LS_API_KEY},
        timeout=10,
    )
    if resp.ok:
        access_token = resp.json().get("access")
        session.headers["Authorization"] = f"Bearer {access_token}"
    else:
        session.headers["Authorization"] = f"Token {LS_API_KEY}"
    return session


def polygons_to_mask(polygons_xy, height, width) -> np.ndarray:
    """Draw multiple polygon contours onto a single binary mask (union)."""
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for polygon_xy in polygons_xy:
        if len(polygon_xy) < 3:
            continue
        draw.polygon([(float(x), float(y)) for x, y in polygon_xy], fill=255)
    return np.array(mask_img)  # 0 or 255, uint8 — mask2rle thresholds at 128


class YOLOSegBackend(LabelStudioMLBase):
    def __init__(self, project_id=None, **kwargs):
        super().__init__(**kwargs)
        print(f"Loading YOLO model from {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH)

    def _get_label_config(self):
        from_name, to_name, labels = 'brush_label', 'image', []
        if self.parsed_label_config:
            for tag_name, tag_info in self.parsed_label_config.items():
                if tag_info.get('type', '').lower() == 'brushlabels':
                    from_name = tag_name
                    to_name = tag_info.get('to_name', ['image'])[0]
                    labels = tag_info.get('labels', [])
                    break
        print(f"[YOLO] from_name={from_name}, labels={labels}")
        return from_name, to_name, labels

    def predict(self, tasks, context=None, **kwargs):
        from_name, to_name, labels = self._get_label_config()
        label_set = {lbl.lower() for lbl in labels} if labels else None
        predictions = []

        for task in tasks:
            image_url = task['data']['image']

            if image_url.startswith('s3://'):
                session = get_ls_session()
                resp = session.get(f"{LS_URL}/api/tasks/{task['id']}/?full=true", timeout=10)
                resp.raise_for_status()
                image_url = resp.json()['data']['image']
                image_url = image_url.replace('http://localhost:8080', LS_URL)
                print(f"[SAM3] Resolved presigned URL: {image_url}")

            elif not image_url.startswith('http'):
                image_url = f"{LS_URL}{image_url}"
            session = get_ls_session()
            resp = session.get(image_url, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            width, height = image.size

            results = self.model.predict(image, conf=CONF_THRESHOLD, verbose=False)
            result_list = []

            if results and results[0].masks is not None:
                r = results[0]
                # Group polygon masks by label name, then merge each into one brush region
                label_polygons: dict[str, list] = {}
                for polygon_xy, cls_idx in zip(r.masks.xy, r.boxes.cls.cpu().numpy()):
                    class_name = r.names[int(cls_idx)]
                    if label_set and class_name.lower() not in label_set:
                        continue
                    label_name = (
                        next((lbl for lbl in labels if lbl.lower() == class_name.lower()), class_name)
                        if labels else class_name
                    )
                    label_polygons.setdefault(label_name, []).append(polygon_xy)

                for label_name, polygons in label_polygons.items():
                    mask_np = polygons_to_mask(polygons, height, width)
                    rle = mask2rle(mask_np)
                    result_list.append({
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
                    })

                print(f"[YOLO] {len(result_list)} brush regions (merged by label)")

            predictions.append({"result": result_list})

        return predictions
