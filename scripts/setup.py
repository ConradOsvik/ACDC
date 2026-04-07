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
def parse_ml_backends(env_value: str) -> list[dict]:
    """
    Parse ML_BACKENDS env var into a list of backend dicts.
    Format: comma-separated entries of  title=url  or  title=url:interactive
    Example: SAM3=http://sam3:9090:interactive,YOLO-seg=http://yolo-seg:9090
    """
    backends = []
    for entry in env_value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise SystemExit(f"Invalid ML_BACKENDS entry '{entry}' — expected format: title=url or title=url:interactive")
        title, rest = entry.split("=", 1)
        interactive = rest.endswith(":interactive")
        url = rest.removesuffix(":interactive")
        backends.append({"title": title.strip(), "url": url.strip(), "is_interactive": interactive})
    return backends

ML_BACKENDS = parse_ml_backends(
    os.environ.get("ML_BACKENDS", "SAM3=http://sam3:9090:interactive,YOLO-seg=http://yolo-seg:9090")
)

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
    for backend in ML_BACKENDS:
        if backend["url"] in existing_backends:
            print(f"backend '{backend['title']}' already connected, skipping")
            continue
        ls.ml.create(project=project.id, url=backend["url"], title=backend["title"], is_interactive=backend["is_interactive"])
        print(f"connected {backend['title']} at {backend['url']}{' (interactive)' if backend['is_interactive'] else ''}")


if __name__ == "__main__":
    main()
