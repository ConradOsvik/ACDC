import os
import shutil
import sys
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path

import label_studio_ml.model as _ls_ml_model
from label_studio_ml.model import LabelStudioMLBase
from PIL import Image
from ultralytics import YOLO

from utils import (
    LS_URL,
    export_annotations_to_yolo,
    get_brush_label_config,
    get_image_path,
    ls_get,
    polygons_to_brush_result,
    polygons_to_mask,
)


# ─── Train-in-background patch ───────────────────────────────────────────────
# The label-studio-ml SDK's SimpleJobManager runs training jobs in a forked
# daemon multiprocess. With PyTorch loaded, fork() inherits CUDA/OMP/file-
# descriptor state that doesn't survive cleanly — the child silently dies and
# its stdout doesn't make it to docker logs.
#
# Replace with a daemon thread: same process (output goes to docker logs), no
# fork hazards, webhook returns immediately so LS UI doesn't time out on long
# trainings. A non-blocking lock prevents two trainings from running at once.
_training_lock = threading.Lock()


def _run_job_threaded(self, model_class, args):
    def _target():
        if not _training_lock.acquire(blocking=False):
            print("[YOLO] Training already in progress; ignoring duplicate request", flush=True)
            return
        try:
            self.job(model_class, *args)
        except Exception:
            print("[YOLO fit] Background training thread raised:", file=sys.stderr, flush=True)
            traceback.print_exc()
        finally:
            _training_lock.release()

    threading.Thread(target=_target, daemon=True, name="yolo-training").start()
    print("[YOLO] Training thread started", flush=True)


_ls_ml_model.SimpleJobManager.run_job = _run_job_threaded


WEIGHTS_DIR = Path(os.environ.get("YOLO_WEIGHTS_DIR", "/data/weights"))
DEFAULT_MODEL = os.environ.get("YOLO_DEFAULT_MODEL", "yolo26m-seg.pt")
CONF_THRESHOLD = float(os.environ.get("YOLO_CONF", "0.25"))
TRAIN_EPOCHS = int(os.environ.get("YOLO_TRAIN_EPOCHS", "50"))
TRAIN_IMGSZ = int(os.environ.get("YOLO_TRAIN_IMGSZ", "640"))
TRAIN_BATCH = int(os.environ.get("YOLO_TRAIN_BATCH", "-1"))


def load_model() -> YOLO:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(WEIGHTS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        print(f"[YOLO] Loading weights from {candidates[0]}", flush=True)
        return YOLO(str(candidates[0]))
    print(f"[YOLO] No weights in {WEIGHTS_DIR}, downloading {DEFAULT_MODEL}", flush=True)
    model = YOLO(DEFAULT_MODEL)
    src = Path(DEFAULT_MODEL)
    if src.exists():
        dest = WEIGHTS_DIR / DEFAULT_MODEL
        shutil.move(str(src), dest)
        print(f"[YOLO] Saved downloaded weights to {dest}", flush=True)
    return model


def _resolve_project_id(data: dict) -> int | None:
    """Pull the project id out of the webhook payload, with an API fallback."""
    project = (data or {}).get("project") if isinstance(data, dict) else None
    if isinstance(project, dict) and project.get("id"):
        return project["id"]
    try:
        resp = ls_get(f"{LS_URL}/api/projects/", timeout=10).json()
        results = resp.get("results", []) if isinstance(resp, dict) else resp
        if isinstance(results, list) and len(results) == 1:
            return results[0]["id"]
    except Exception as e:
        print(f"[YOLO fit] Could not list projects via LS API: {e}", flush=True)
    return None


class YOLOSegBackend(LabelStudioMLBase):
    TRAIN_EVENTS = LabelStudioMLBase.TRAIN_EVENTS + ("START_TRAINING",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = load_model()
        print("[YOLO] Model loaded", flush=True)

    def _get_result_from_job_id(self, _job_id):
        return {}

    def predict(self, tasks, context=None, **kwargs):
        from_name, to_name, labels = get_brush_label_config(self.parsed_label_config, "brushlabels")
        print(f"[YOLO predict] from_name={from_name}, labels={labels}", flush=True)
        label_set = {lbl.lower() for lbl in labels} if labels else None
        predictions = []

        for task in tasks:
            local_path = get_image_path(task["data"]["image"], task_id=task["id"])
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

                print(f"[YOLO predict] {len(result_list)} brush regions", flush=True)

            predictions.append({"result": result_list})

        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        # SDK passes the webhook payload as kwargs['data']; older callers pass a
        # list of completions where each item carries 'project'. Try both.
        project_id = None
        if completions and isinstance(completions, list) and isinstance(completions[0], dict):
            project_id = completions[0].get("project")
        if not project_id:
            project_id = _resolve_project_id(kwargs.get("data") or {})
        if not project_id:
            print("[YOLO fit] Could not determine project id; skipping training", flush=True)
            return {}

        print(
            f"[YOLO fit] Starting training for project {project_id} "
            f"(epochs={TRAIN_EPOCHS}, imgsz={TRAIN_IMGSZ}, batch={TRAIN_BATCH})",
            flush=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            try:
                classes, exported, skipped = export_annotations_to_yolo(project_id, dataset_dir)
            except ValueError as e:
                # Configuration / dataset-shape problem (no labels, too few tasks, etc.)
                # These are user-actionable; surface them clearly and stop.
                print(f"[YOLO fit] Cannot train: {e}", flush=True)
                return {}

            print(f"[YOLO fit] Exported {exported} tasks ({skipped} skipped)", flush=True)
            if exported < 2:
                print(
                    f"[YOLO fit] Cannot train: need at least 2 tasks with brush annotations "
                    f"(have {exported}). Annotate more, then click Start Training again.",
                    flush=True,
                )
                return {}

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"train_{timestamp}"
            results = self.model.train(
                data=str(dataset_dir / "data.yaml"),
                task="segment",
                epochs=TRAIN_EPOCHS,
                imgsz=TRAIN_IMGSZ,
                batch=TRAIN_BATCH,
                workers=0,
                plots=True,         # Writes results.png, confusion_matrix.png, P/R/F1/PR_curve.png, val_batch*.jpg
                verbose=True,
                project="/data/runs",
                name=run_name,
                exist_ok=False,
            )

            best = Path(results.save_dir) / "weights" / "best.pt"
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            dest = WEIGHTS_DIR / f"yolo_{timestamp}.pt"
            shutil.copy(best, dest)
            print(f"[YOLO fit] Saved new weights to {dest}", flush=True)
            print(f"[YOLO fit] Run artifacts: /data/runs/{run_name}/ (host: runs/yolo/{run_name}/)", flush=True)

        self.model = YOLO(str(dest))
        print("[YOLO fit] Training complete; model reloaded.", flush=True)
        return {"model_path": str(dest)}
