import io
import os
import tempfile
import uuid
import requests
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from ultralytics.models.sam import SAM3SemanticPredictor

LS_URL = os.environ.get('LABEL_STUDIO_URL', 'http://localhost:8080')
LS_API_KEY = os.environ.get('LABEL_STUDIO_API_KEY', '')
MODEL_PATH = os.environ.get('SAM3_MODEL_PATH', 'sam3.pt')


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
        from_name, to_name, labels = 'label', 'image', []
        if self.parsed_label_config:
            for tag_name, tag_info in self.parsed_label_config.items():
                if tag_info.get('type', '').lower().endswith('labels'):
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
            # Fetch image from Label Studio
            image_url = task['data']['image']
            if not image_url.startswith('http'):
                image_url = f"{LS_URL}{image_url}"
            print(f"[SAM3] Fetching image: {image_url}")
            session = get_ls_session()
            resp = session.get(image_url, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            width, height = image.size
            print(f"[SAM3] Image size: {width}x{height}")

            # Save to temp file — same as run_ultralytics.py uses a file path
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                image.save(tmp_path)

            try:
                self.predictor.set_image(tmp_path)
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
                    print(f"[SAM3] Raw results: {results}")
                    if results and results[0].masks is not None:
                        print(f"[SAM3] Masks count: {len(results[0].masks.xy)}")
                        result_list.extend(
                            self._masks_to_polygons(results[0].masks.xy, width, height, label_name, from_name, to_name)
                        )
                else:
                    query_labels = labels if labels else ["object"]
                    print(f"[SAM3] Text mode, querying: {query_labels}")
                    for label_name in query_labels:
                        results = self.predictor(text=[label_name.lower()])
                        print(f"[SAM3] '{label_name}' raw results: {results}")
                        if results and results[0].masks is not None:
                            n = len(results[0].masks.xy)
                            print(f"[SAM3] '{label_name}': {n} masks")
                            result_list.extend(
                                self._masks_to_polygons(results[0].masks.xy, width, height, label_name, from_name, to_name)
                            )
                        else:
                            print(f"[SAM3] '{label_name}': no masks returned")
            finally:
                os.unlink(tmp_path)

            print(f"[SAM3] Returning {len(result_list)} polygons")
            predictions.append({"result": result_list})

        return predictions

    def _masks_to_polygons(self, masks_xy, width, height, label_name, from_name, to_name):
        result_list = []
        for polygon_xy in masks_xy:
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
        return result_list
