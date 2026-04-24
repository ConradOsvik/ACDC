"""
Idempotently creates the Label Studio project, connects MinIO storage, and
registers the ML backends. Run once after Label Studio is up.

    uv run scripts/setup.py
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio

load_dotenv(Path(__file__).parent / ".env")

LABEL_STUDIO_URL = os.environ["LABEL_STUDIO_URL"]
LABEL_STUDIO_API_KEY = os.environ["LABEL_STUDIO_API_KEY"]
if not LABEL_STUDIO_API_KEY:
    raise SystemExit("LABEL_STUDIO_API_KEY is not set — get it from Label Studio UI → Account & Settings → Access Token")
ML_BACKEND_URL = os.environ.get("ML_BACKEND_URL", "http://sam3:9090")
ML_BACKEND_TITLE = os.environ.get("ML_BACKEND_TITLE", "SAM3")
ML_BACKEND_INTERACTIVE = os.environ.get("ML_BACKEND_INTERACTIVE", "true").lower() == "true"

PROJECT_TITLE = "Corrosion Annotation"

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true"/>

  <BrushLabels name="brush_label" toName="image">
    <Label value="Red Rust"    background="#FF6B00"/>
    <Label value="White Rust"  background="#00AAFF"/>
  </BrushLabels>

  <RectangleLabels name="bbox" toName="image" smart="true" showInline="true">
    <Label value="Red Rust"    background="#8B2500"/>
    <Label value="White Rust"  background="#005580"/>
  </RectangleLabels>
</View>"""

def main():
    ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

    existing = next((p for p in ls.projects.list() if p.title == PROJECT_TITLE), None)
    if existing:
        print(f"project '{PROJECT_TITLE}' already exists (id={existing.id})")
        project = existing
    else:
        project = ls.projects.create(title=PROJECT_TITLE, label_config=LABEL_CONFIG)
        print(f"created project '{PROJECT_TITLE}' (id={project.id})")

    existing_backends = {b.url for b in ls.ml.list(project=project.id)}
    if ML_BACKEND_URL in existing_backends:
        print(f"backend '{ML_BACKEND_TITLE}' already connected, skipping")
    else:
        ls.ml.create(project=project.id, url=ML_BACKEND_URL, title=ML_BACKEND_TITLE, is_interactive=ML_BACKEND_INTERACTIVE)
        print(f"connected {ML_BACKEND_TITLE} at {ML_BACKEND_URL}{' (interactive)' if ML_BACKEND_INTERACTIVE else ''}")


if __name__ == "__main__":
    main()
