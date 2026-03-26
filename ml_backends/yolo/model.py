import os
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from utils import get_image_path, get_brush_label_config, polygons_to_mask, polygons_to_brush_result

MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', 'yolo11n-seg.pt')
CONF_THRESHOLD = float(os.environ.get('YOLO_CONF', '0.25'))


class YOLOSegBackend(LabelStudioMLBase):
    def __init__(self, project_id=None, **kwargs):
        super().__init__(**kwargs)
        print(f"Loading YOLO model from {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH)

    def predict(self, tasks, context=None, **kwargs):
        from_name, to_name, labels = get_brush_label_config(self.parsed_label_config, 'brushlabels')
        print(f"[YOLO] from_name={from_name}, labels={labels}")
        label_set = {lbl.lower() for lbl in labels} if labels else None
        predictions = []

        for task in tasks:
            local_path = get_image_path(task['data']['image'], task_id=task['id'])
            print(f"[YOLO] Loading: {local_path}")
            image = Image.open(local_path).convert("RGB")
            width, height = image.size

            results = self.model.predict(image, conf=CONF_THRESHOLD, verbose=False)
            result_list = []

            if results and results[0].masks is not None:
                r = results[0]
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
                    result_list.append(
                        polygons_to_brush_result(polygons, height, width, label_name, from_name, to_name)
                    )

                print(f"[YOLO] {len(result_list)} brush regions (merged by label)")

            predictions.append({"result": result_list})

        return predictions
