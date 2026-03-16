"""
Idempotently creates the Label Studio project, connects MinIO storage, and
registers the ML backends. Run once after Label Studio is up.

    uv run scripts/setup.py
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from label_studio_sdk.client import LabelStudio
from minio import Minio
from minio.error import S3Error

load_dotenv(Path(__file__).parent / ".env")

LABEL_STUDIO_URL = os.environ["LABEL_STUDIO_URL"]
LABEL_STUDIO_API_KEY = os.environ["LABEL_STUDIO_API_KEY"]
if not LABEL_STUDIO_API_KEY:
    raise SystemExit("LABEL_STUDIO_API_KEY is not set — get it from Label Studio UI → Account & Settings → Access Token")
SAM3_URL = os.environ.get("SAM3_URL", "http://sam3:9090")
YOLO_URL = os.environ.get("YOLO_URL", "http://yolo-seg:9090")
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
# Endpoint LS uses to reach MinIO from inside Docker (service name, not localhost)
MINIO_LS_ENDPOINT = os.environ.get("MINIO_LS_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "labelstudio")

PROJECT_TITLE = "Corrosion Annotation"

LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true"/>

  <BrushLabels name="brush_label" toName="image">
    <Label value="Red Rust"    background="#FF6B00"/>
    <Label value="White Rust"  background="#00AAFF"/>
    <Label value="Blistering"  background="#FF0000"/>
    <Label value="Flaking"     background="#FFD700"/>
  </BrushLabels>

  <RectangleLabels name="bbox" toName="image" smart="true" showInline="true">
    <Label value="Red Rust"    background="#8B2500"/>
    <Label value="White Rust"  background="#005580"/>
    <Label value="Blistering"  background="#7B0000"/>
    <Label value="Flaking"     background="#8B7000"/>
  </RectangleLabels>
</View>"""


def ensure_bucket():
    host = MINIO_ENDPOINT.removeprefix("https://").removeprefix("http://")
    secure = MINIO_ENDPOINT.startswith("https://")
    client = Minio(host, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=secure)
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
            print(f"created bucket: {MINIO_BUCKET}")
        else:
            print(f"bucket exists: {MINIO_BUCKET}")
    except S3Error as e:
        print(f"minio error: {e}", file=sys.stderr)
        raise


def main():
    ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

    existing = next((p for p in ls.projects.list() if p.title == PROJECT_TITLE), None)
    if existing:
        print(f"project '{PROJECT_TITLE}' already exists (id={existing.id}), skipping")
        return

    ensure_bucket()

    project = ls.projects.create(title=PROJECT_TITLE, label_config=LABEL_CONFIG)
    print(f"created project '{PROJECT_TITLE}' (id={project.id})")

    # SDK's s3.create() doesn't expose s3_endpoint, so call the API directly.
    # Reuse the SDK's auth headers (handles both legacy Token and JWT Bearer formats).
    resp = requests.post(
        f"{LABEL_STUDIO_URL}/api/storages/s3/",
        headers=ls._client_wrapper.get_headers(),
        json={
            "project": project.id,
            "title": "MinIO Images",
            "bucket": MINIO_BUCKET,
            "use_blob_urls": True,
            "aws_access_key_id": MINIO_ACCESS_KEY,
            "aws_secret_access_key": MINIO_SECRET_KEY,
            "s3_endpoint": MINIO_LS_ENDPOINT,
            "region_name": "us-east-1",
            "presign": False,
        },
    )
    resp.raise_for_status()
    storage = resp.json()
    print(f"connected MinIO storage (bucket={MINIO_BUCKET}, presign={storage.get('presign')}, presign_ttl={storage.get('presign_ttl')})")

    ls.ml.create(project=project.id, url=SAM3_URL, title="SAM3", is_interactive=True)
    print(f"connected SAM3 at {SAM3_URL}")

    ls.ml.create(project=project.id, url=YOLO_URL, title="YOLO-seg")
    print(f"connected YOLO-seg at {YOLO_URL}")


if __name__ == "__main__":
    main()
