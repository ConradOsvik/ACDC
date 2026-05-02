"""Centralized configuration loaded from environment / .env files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path | None:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def _find_env_file() -> Path | None:
    """Look for .env in cwd, then <project_root>/.env."""
    cwd = Path.cwd() / ".env"
    if cwd.is_file():
        return cwd
    root = _project_root()
    if root and (root / ".env").is_file():
        return root / ".env"
    return None


@dataclass(frozen=True)
class Settings:
    label_studio_url: str
    label_studio_api_key: str
    sam3_url: str = "http://sam3:9090"
    yolo_url: str = "http://yolo-seg:9090"
    yolo_conf: float = 0.25
    compose_ui_file: str = ""
    compose_models_file: str = ""
    env_file: str = ""

    @staticmethod
    def load(env_file: str | None = None, *, require_api_key: bool = True) -> Settings:
        path = Path(env_file) if env_file else _find_env_file()
        if path and path.is_file():
            load_dotenv(path, override=True)

        api_key = os.environ.get("LABEL_STUDIO_API_KEY", "")
        if require_api_key and not api_key:
            from ls_cli.utils.output import error

            error(
                "LABEL_STUDIO_API_KEY is not set.\n"
                "Get it from Label Studio UI → Account & Settings → Access Token, "
                "then set it in your .env file."
            )
            raise SystemExit(1)

        root = _project_root()
        compose_ui = compose_models = ""
        if root:
            compose_ui = str(root / "infra" / "dev" / "docker-compose.ui.yml")
            compose_models = str(root / "infra" / "dev" / "docker-compose.models.yml")

        return Settings(
            label_studio_url=os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080"),
            label_studio_api_key=api_key,
            sam3_url=os.environ.get("SAM3_URL", "http://sam3:9090"),
            yolo_url=os.environ.get("YOLO_URL", "http://yolo-seg:9090"),
            yolo_conf=float(os.environ.get("YOLO_CONF", "0.25")),
            compose_ui_file=os.environ.get("COMPOSE_UI_FILE", compose_ui),
            compose_models_file=os.environ.get("COMPOSE_MODELS_FILE", compose_models),
            env_file=str(path) if path else "",
        )
