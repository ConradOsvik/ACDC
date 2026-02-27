import os
from label_studio_sdk import LabelStudio
from dotenv import load_dotenv

load_dotenv()

def main():
    LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL")
    LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

    if not LABEL_STUDIO_URL or not LABEL_STUDIO_API_KEY:
        raise ValueError("LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY must be set")

    client = LabelStudio(
        base_url=LABEL_STUDIO_URL,
        api_key=LABEL_STUDIO_API_KEY
    )

    projects = client.projects.list().items
    print(projects)

    if projects:
        project_id = projects[0].id

        backends = [
            {"url": "http://localhost:9090", "title": "YOLO-seg"},
            {"url": "http://localhost:9091", "title": "MobileSAM", "is_interactive": True},
            {"url": "http://localhost:9092", "title": "SAM 3"},
        ]

        for backend in backends:
            ml = client.ml.create(project=project_id, **backend)
            print(f"Connected {ml.title} (id={ml.id})")


if __name__ == "__main__":
    main()
