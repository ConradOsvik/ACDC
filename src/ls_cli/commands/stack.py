"""Stack lifecycle: bring services up/down, bootstrap a project end-to-end."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer

from ls_cli.utils.docker import compose_down, compose_up
from ls_cli.utils.output import error, info, success


def _models_compose_file(env_file: str | None, gpu: bool) -> str:
    """Resolve the models compose file, swapping in the GPU variant if requested."""
    from ls_cli.config import Settings, _project_root

    settings = Settings.load(env_file, require_api_key=False)
    if not gpu:
        return settings.compose_models_file

    root = _project_root()
    if not root:
        error("Could not locate project root.")
        raise typer.Exit(1)
    return str(root / "infra" / "dev" / "docker-compose.models.gpu.yml")


def _ensure_dotenv(env_file: str | None) -> Path:
    """Make sure a .env file exists. If not, copy from the root .env.example."""
    from ls_cli.config import _find_env_file, _project_root

    if env_file:
        path = Path(env_file)
        if path.is_file():
            return path

    found = _find_env_file()
    if found:
        return found

    root = _project_root()
    if not root:
        error("Could not locate project root.")
        raise typer.Exit(1)

    example = root / ".env.example"
    target = root / ".env"
    if not example.is_file():
        error(f"No .env found and {example} doesn't exist either.")
        raise typer.Exit(1)
    target.write_text(example.read_text())
    info(f"Created {target} from {example.name} — set LABEL_STUDIO_API_KEY in it.")
    return target


def _wait_for_label_studio(url: str, timeout_s: int = 180) -> bool:
    """Poll LS /health until 2xx, or timeout. (LS isn't easily covered by
    `compose up --wait` because we want to validate the API token before
    bringing up dependent services.)"""
    import requests

    deadline = time.time() + timeout_s
    last_err: str | None = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url.rstrip('/')}/health", timeout=5)
            if resp.ok:
                return True
            last_err = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(2)
    if last_err:
        info(f"  last error: {last_err}")
    return False


def _token_works(settings) -> bool:
    """Hit /api/projects/ once to check whether the current API key is valid."""
    if not settings.label_studio_api_key:
        return False
    try:
        from ls_cli.client import get_client

        client = get_client(settings)
        # Trigger a single API call. SDK auto-refreshes JWTs, so this surfaces
        # any auth issue immediately.
        list(client.projects.list())
        return True
    except Exception:
        return False


def _prompt_for_token(env_path: Path, base_url: str) -> str:
    """Walk the user through grabbing a fresh API token and writing it to .env."""
    info("")
    info("Get your API token:")
    info(f"  1. Open {base_url} in a browser")
    info("  2. Sign in (default user: admin@example.com / changeme — see your .env)")
    info("  3. Click your avatar → Account & Settings → Access Token → Copy")
    info("")
    token = typer.prompt("Paste the token here", hide_input=True).strip()
    if not token:
        error("No token entered.")
        raise typer.Exit(1)

    text = env_path.read_text() if env_path.is_file() else ""
    lines = text.splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith("LABEL_STUDIO_API_KEY="):
            lines[i] = f"LABEL_STUDIO_API_KEY={token}"
            found = True
            break
    if not found:
        lines.append(f"LABEL_STUDIO_API_KEY={token}")
    env_path.write_text("\n".join(lines) + "\n")
    info(f"Wrote token to {env_path}")
    return token


def up(
    gpu: bool = typer.Option(False, help="Use the GPU compose variant for the model containers"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Bring the full stack up (Label Studio + Postgres + ML backends)."""
    from ls_cli.config import Settings

    _ensure_dotenv(env_file)
    settings = Settings.load(env_file, require_api_key=False)
    models = _models_compose_file(env_file, gpu)

    if not settings.compose_ui_file:
        error("Could not locate the UI compose file.")
        raise typer.Exit(1)

    info(f"Starting stack ({'GPU' if gpu else 'CPU'} models)...")
    compose_up(settings.compose_ui_file, models, env_file=settings.env_file)
    success(f"Stack up. Label Studio: {settings.label_studio_url}")


def down(
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Stop the full stack."""
    from ls_cli.config import Settings

    settings = Settings.load(env_file, require_api_key=False)
    if not settings.compose_ui_file or not settings.compose_models_file:
        error("Could not locate compose files.")
        raise typer.Exit(1)
    compose_down(settings.compose_ui_file, settings.compose_models_file, env_file=settings.env_file)
    success("Stack stopped.")


def init(
    title: str = typer.Option("Corrosion Annotation", help="Project title"),
    gpu: bool = typer.Option(False, help="Use the GPU compose variant for the model containers"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Bring the stack up, wait for it, and create a project + YOLO backend.

    The first time you run this, you'll need to grab an API token from the
    Label Studio UI and paste it into the .env file. The command will pause
    and walk you through it.
    """
    from ls_cli.client import get_client
    from ls_cli.commands.project import DEFAULT_LABEL_CONFIG
    from ls_cli.config import Settings
    from ls_cli.utils.backends import BackendChoice, ensure_single_backend

    env_path = _ensure_dotenv(env_file)
    settings = Settings.load(env_file, require_api_key=False)
    models = _models_compose_file(env_file, gpu)

    info("[1/5] Starting Label Studio...")
    compose_up(settings.compose_ui_file, env_file=settings.env_file)

    info(f"[2/5] Waiting for Label Studio at {settings.label_studio_url}...")
    if not _wait_for_label_studio(settings.label_studio_url):
        error("Label Studio did not become healthy in time. Check 'docker compose logs'.")
        raise typer.Exit(1)
    success("Label Studio is up.")

    info("[3/5] Validating API token...")
    attempts = 0
    while not _token_works(settings):
        if attempts == 0 and settings.label_studio_api_key:
            info("Existing token is invalid (likely stale after a data wipe).")
        attempts += 1
        if attempts > 3:
            error("Too many failed attempts. Aborting.")
            raise typer.Exit(1)
        _prompt_for_token(env_path, settings.label_studio_url)
        settings = Settings.load(env_file, require_api_key=False)
    success("Token works.")

    info(f"[4/5] Starting model containers ({'GPU' if gpu else 'CPU'})...")
    info("(this can take a minute on first boot — model weights need to download)")
    compose_up(models, env_file=settings.env_file, recreate=True, wait=True)
    success("Model backends are healthy.")

    info("[5/5] Creating project and attaching YOLO backend...")
    ls = get_client(settings)
    existing = next((p for p in ls.projects.list() if p.title == title), None)
    if existing:
        project = existing
        info(f"Project '{title}' already exists (id={project.id}), reusing")
    else:
        project = ls.projects.create(title=title, label_config=DEFAULT_LABEL_CONFIG)
        success(f"Created project '{title}' (id={project.id})")

    ensure_single_backend(ls, settings, project.id, BackendChoice.yolo)

    info("")
    success(f"Ready. Open {settings.label_studio_url} and start labelling.")
    info("Tips:")
    info(f"  - Switch backend:   ls-cli backend switch {project.id} sam3")
    info(f"  - Train YOLO:       click 'Start Training' in the LS UI, or: ls-cli yolo train")
    info("  - Deploy weights:   ls-cli yolo deploy path/to/best.pt")
    info("  - Stop the stack:   ls-cli down")
