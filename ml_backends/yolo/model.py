import os
import shutil
import tempfile
from pathlib import Path
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO
from utils import LS_URL, ls_get, get_image_path, get_brush_label_config, polygons_to_mask, polygons_to_brush_result, export_annotations_to_yolo

MODEL_PATH = Path(os.environ.get('YOLO_MODEL_PATH', 'yolo26m-seg.pt'))
CONF_THRESHOLD = float(os.environ.get('YOLO_CONF', '0.25'))
TRAIN_EPOCHS = int(os.environ.get('YOLO_TRAIN_EPOCHS', '50'))
TRAIN_IMGSZ = int(os.environ.get('YOLO_TRAIN_IMGSZ', '640'))


class YOLOSegBackend(LabelStudioMLBase):
    TRAIN_EVENTS = LabelStudioMLBase.TRAIN_EVENTS + ('START_TRAINING',)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Loading YOLO model from {MODEL_PATH}...")
        try:
            self.model = YOLO(str(MODEL_PATH))
            print(f"[YOLO] Model loaded successfully")
        except Exception as e:
            print(f"[YOLO] ERROR loading model: {e}")
            raise

    def _get_result_from_job_id(self, _job_id):
        return {}

    def _ensure_model_loaded(self):
        if not hasattr(self, 'model') or self.model is None:
            print(f"[YOLO] Model not loaded, loading from {MODEL_PATH}...")
            self.model = YOLO(str(MODEL_PATH))

    def predict(self, tasks, context=None, **kwargs):
        self._ensure_model_loaded()
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

    def fit(self, completions, workdir=None, **kwargs):
        project_id = None
        if completions and isinstance(completions, list) and isinstance(completions[0], dict):
            project_id = completions[0].get('project')
        if not project_id:
            try:
                projects = ls_get(f'{LS_URL}/api/projects/').json().get('results', [])
                if len(projects) == 1:
                    project_id = projects[0]['id']
            except Exception as e:
                print(f"[YOLO fit] Could not determine project_id: {e}")
        if not project_id:
            print("[YOLO fit] Cannot determine project_id, skipping training")
            return {}

        print(f"[YOLO fit] Starting training for project {project_id} ({TRAIN_EPOCHS} epochs)...")

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / 'dataset'
            try:
                classes, exported, skipped = export_annotations_to_yolo(project_id, dataset_dir)
            except ValueError as e:
                print(f"[YOLO fit] Skipping training: {e}")
                return {}

            print(f"[YOLO fit] Exported {exported} tasks, skipped {skipped} unannotated")
            if exported == 0:
                print("[YOLO fit] No annotated tasks found, skipping training")
                return {}

            results = self.model.train(
                data=str(dataset_dir / 'data.yaml'),
                task='segment',
                epochs=TRAIN_EPOCHS,
                imgsz=TRAIN_IMGSZ,
                workers=0,
                plots=False,
            )

            best = Path(results.save_dir) / 'weights' / 'best.pt'
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(best, MODEL_PATH)

        self.model = YOLO(str(MODEL_PATH))
        print(f"[YOLO fit] Training complete. Model saved to {MODEL_PATH}")
        return {'model_path': str(MODEL_PATH)}
