import io
import os
import uuid
import requests
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO

LS_URL = os.environ.get('LABEL_STUDIO_URL', 'http://localhost:8080')
LS_API_KEY = os.environ.get('LABEL_STUDIO_API_KEY', '')
MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', 'yolo11n-seg.pt')
CONF_THRESHOLD = float(os.environ.get('YOLO_CONF', '0.25'))


def get_ls_session():
    """Get an authenticated requests session for Label Studio."""
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


class YOLOSegBackend(LabelStudioMLBase):
    def __init__(self, project_id=None, **kwargs):
        super().__init__(**kwargs)
        print(f"Loading YOLO model from {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH)

    def _get_label_config(self):
        """Parse label names from the project config. Called on every predict."""
        from_name, to_name, labels = 'label', 'image', []
        if self.parsed_label_config:
            print(f"[YOLO] parsed_label_config keys: {list(self.parsed_label_config.keys())}")
            for tag_name, tag_info in self.parsed_label_config.items():
                tag_type = tag_info.get('type', '').lower()
                if tag_type.endswith('labels'):
                    from_name = tag_name
                    to_name = tag_info.get('to_name', ['image'])[0]
                    labels = tag_info.get('labels', [])
                    break
        print(f"[YOLO] Using from_name={from_name}, labels={labels}")
        return from_name, to_name, labels

    def predict(self, tasks, context=None, **kwargs):
        from_name, to_name, labels = self._get_label_config()
        label_set = {lbl.lower() for lbl in labels} if labels else None
        predictions = []

        for task in tasks:
            # 1. Fetch the image from Label Studio
            image_url = task['data']['image']
            if not image_url.startswith('http'):
                image_url = f"{LS_URL}{image_url}"
            session = get_ls_session()
            resp = session.get(image_url, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            width, height = image.size

            # 2. Run YOLO segmentation
            results = self.model.predict(image, conf=CONF_THRESHOLD, verbose=False)
            result_list = []

            if results and results[0].masks:
                r = results[0]
                for polygon_xy, cls_idx in zip(r.masks.xy, r.boxes.cls.cpu().numpy()):
                    class_name = r.names[int(cls_idx)]

                    if label_set and class_name.lower() not in label_set:
                        continue

                    # Preserve the LS label casing if configured, else use YOLO class name
                    if labels:
                        label_name = next(
                            (lbl for lbl in labels if lbl.lower() == class_name.lower()),
                            class_name,
                        )
                    else:
                        label_name = class_name

                    if len(polygon_xy) < 3:
                        continue

                    points = [
                        [float(x) / width * 100, float(y) / height * 100]
                        for x, y in polygon_xy
                    ]
                    result_list.append({
                        "id": uuid.uuid4().hex[:8],
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "polygonlabels",
                        "value": {
                            "points": points,
                            "polygonlabels": [label_name],
                        }
                    })

                print(f"[YOLO] {len(result_list)} detections returned")

            predictions.append({"result": result_list})

        return predictions
