import os
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from ultralytics.models.sam import SAM3SemanticPredictor
from utils import get_image_path, get_brush_label_config, polygons_to_brush_result

MODEL_PATH = os.environ.get('SAM3_MODEL_PATH', 'sam3.pt')


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

    def predict(self, tasks, context=None, **kwargs):
        from_name, to_name, labels = get_brush_label_config(self.parsed_label_config, 'brushlabels')
        print(f"[SAM3] from_name={from_name}, labels={labels}")
        predictions = []

        for task in tasks:
            local_path = get_image_path(task['data']['image'], task_id=task['id'])
            print(f"[SAM3] Loading: {local_path}")
            image = Image.open(local_path).convert("RGB")
            width, height = image.size

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
                        polygons_to_brush_result(polygons, height, width, label_name, from_name, to_name)
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
                            polygons_to_brush_result(polygons, height, width, label_name, from_name, to_name)
                        )
                    else:
                        print(f"[SAM3] '{label_name}': no masks")

            print(f"[SAM3] Returning {len(result_list)} brush regions")
            predictions.append({"result": result_list})

        return predictions
