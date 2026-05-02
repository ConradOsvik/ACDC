"""Backend selection helpers — enforce a single ML backend per project."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from ls_cli.config import Settings
from ls_cli.utils.output import info, success

if TYPE_CHECKING:
    from label_studio_sdk.client import LabelStudio


class BackendChoice(str, Enum):
    yolo = "yolo"
    sam3 = "sam3"
    none = "none"


_BACKEND_SPECS = {
    BackendChoice.yolo: {"title": "YOLO-seg", "is_interactive": False},
    BackendChoice.sam3: {"title": "SAM3", "is_interactive": True},
}


def _backend_url(settings: Settings, choice: BackendChoice) -> str:
    if choice is BackendChoice.yolo:
        return settings.yolo_url
    if choice is BackendChoice.sam3:
        return settings.sam3_url
    raise ValueError(f"No URL for backend choice {choice}")


def ensure_single_backend(
    ls: LabelStudio,
    settings: Settings,
    project_id: int,
    choice: BackendChoice,
) -> None:
    """Make the project's ML backends match `choice` exactly.

    Removes any existing backends that don't match, and adds the chosen one
    if it isn't already attached. choice == none removes all backends.
    """
    existing = list(ls.ml.list(project=project_id))
    spec = _BACKEND_SPECS.get(choice)

    keep_ids = set()
    if spec:
        url = _backend_url(settings, choice)
        match = next((b for b in existing if b.title == spec["title"] and b.url == url), None)
        if match:
            keep_ids.add(match.id)

    for backend in existing:
        if backend.id in keep_ids:
            continue
        ls.ml.delete(id=backend.id)
        info(f"Removed backend '{backend.title}' (id={backend.id})")

    if spec and not keep_ids:
        url = _backend_url(settings, choice)
        ls.ml.create(
            project=project_id,
            url=url,
            title=spec["title"],
            is_interactive=spec["is_interactive"],
        )
        success(f"Connected {spec['title']} at {url}")
    elif spec:
        info(f"{spec['title']} already attached to project {project_id}")
    else:
        info(f"No ML backend attached to project {project_id}")
