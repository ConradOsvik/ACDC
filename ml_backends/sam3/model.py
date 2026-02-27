"""
SAM 3 Label Studio ML Backend

A custom Label Studio ML backend that loads SAM 3 from HuggingFace.
Serves as an extensible template for adding future HuggingFace models.

Supports two modes:
- Batch pre-annotation: uses label names as text prompts for concept segmentation
- Interactive mode: processes keypoint clicks and bounding box prompts via text

Configure the model via the SAM3_MODEL_NAME environment variable.
"""

import os
import logging
from uuid import uuid4

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model

from label_studio_ml.model import LabelStudioMLBase
from label_studio_converter import brush

logger = logging.getLogger(__name__)

SAM3_MODEL_NAME = os.environ.get("SAM3_MODEL_NAME", "facebook/sam3")


class Sam3Backend(LabelStudioMLBase):

    def setup(self):
        """Load the SAM 3 model and processor from HuggingFace."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model {SAM3_MODEL_NAME} on {self.device}")

        self.processor = Sam3Processor.from_pretrained(SAM3_MODEL_NAME)
        self.model = Sam3Model.from_pretrained(SAM3_MODEL_NAME).to(self.device)
        self.model.eval()

        self.set("model_version", f"sam3-{SAM3_MODEL_NAME.split('/')[-1]}")
        logger.info("Model loaded successfully")

    def predict(self, tasks, context=None, **kwargs):
        """Generate mask predictions for the given tasks.

        Batch mode: uses label names from labeling config as text prompts.
        Interactive mode: extracts the label from clicks/boxes and uses it
        as a text prompt.
        """
        results = []
        for task in tasks:
            image = self._load_image(task)
            width, height = image.size
            from_name, to_name, labels = self._get_label_config()

            if context and context.get("result"):
                predictions = self._predict_interactive(
                    image, context, from_name, to_name, width, height
                )
            else:
                predictions = self._predict_batch(
                    image, from_name, to_name, labels, width, height
                )

            results.append(predictions)
        return results

    def _load_image(self, task):
        """Load an image from a task, resolving Label Studio URLs."""
        image_url = task["data"].get(
            self._get_image_value_key(), list(task["data"].values())[0]
        )
        local_path = self.get_local_path(image_url, task_id=task.get("id"))
        return Image.open(local_path).convert("RGB")

    def _get_image_value_key(self):
        """Extract the image data key from the labeling config."""
        for _, tag_info in self.parsed_label_config.items():
            if tag_info.get("type") == "Image":
                return tag_info.get("value", "image")
            for inp in tag_info.get("inputs", []):
                if inp.get("type") == "Image":
                    return inp.get("value", "image")
        return "image"

    def _get_label_config(self):
        """Extract BrushLabels tag info and all label names from the config."""
        from_name, to_name, _ = self.get_first_tag_occurence(
            "BrushLabels", "Image"
        )
        labels = []
        for _, tag_info in self.parsed_label_config.items():
            if tag_info.get("type") == "BrushLabels":
                labels = tag_info.get("labels", [])
                break
        if not labels:
            labels = ["Object"]
        return from_name, to_name, labels

    def _segment_with_text(self, image, text, width, height, threshold=0.5):
        """Run SAM 3 text-prompted segmentation and return masks + scores."""
        inputs = self.processor(
            images=image, text=text, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = [m.cpu().numpy() for m in results["masks"]]
        scores = [float(s) for s in results["scores"]]
        return masks, scores

    def _predict_batch(self, image, from_name, to_name, labels, width, height):
        """Use each label name as a text prompt to find all instances."""
        all_results = []
        total_score = 0.0
        count = 0

        for label in labels:
            masks, scores = self._segment_with_text(
                image, label, width, height
            )
            for mask, score in zip(masks, scores):
                mask_uint8 = (mask > 0).astype(np.uint8) * 255
                rle = brush.mask2rle(mask_uint8)
                all_results.append({
                    "id": str(uuid4())[:4],
                    "from_name": from_name,
                    "to_name": to_name,
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "format": "rle",
                        "rle": rle,
                        "brushlabels": [label],
                    },
                    "score": score,
                    "type": "brushlabels",
                    "readonly": False,
                })
                total_score += score
                count += 1

        return {
            "result": all_results,
            "model_version": self.get("model_version"),
            "score": total_score / max(count, 1),
        }

    def _predict_interactive(self, image, context, from_name, to_name, width, height):
        """Process interactive prompts by extracting the label as a text prompt."""
        label = None

        for result in context["result"]:
            value = result.get("value", {})
            result_type = result.get("type")

            if result_type == "keypointlabels":
                labels = value.get("keypointlabels", [])
                if labels:
                    label = labels[0]
            elif result_type == "rectanglelabels":
                labels = value.get("rectanglelabels", [])
                if labels:
                    label = labels[0]

            if label:
                break

        if not label:
            label = "Object"

        masks, scores = self._segment_with_text(image, label, width, height)

        if not masks:
            return {"result": [], "model_version": self.get("model_version")}

        all_results = []
        total_score = 0.0

        for mask, score in zip(masks, scores):
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            rle = brush.mask2rle(mask_uint8)
            all_results.append({
                "id": str(uuid4())[:4],
                "from_name": from_name,
                "to_name": to_name,
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": [label],
                },
                "score": score,
                "type": "brushlabels",
                "readonly": False,
            })
            total_score += score

        return {
            "result": all_results,
            "model_version": self.get("model_version"),
            "score": total_score / max(len(all_results), 1),
        }
