"""ML backend management commands."""

from __future__ import annotations

from typing import Optional

import typer

from ls_cli.utils.backends import BackendChoice, ensure_single_backend
from ls_cli.utils.output import info, print_table

app = typer.Typer(help="Manage the ML backend attached to a project.")


@app.command("list")
def list_backends(
    project: Optional[int] = typer.Option(None, "--project", "-p", help="Filter by project ID"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """List ML backends across projects (or filtered to one)."""
    from ls_cli.client import get_client
    from ls_cli.config import Settings

    settings = Settings.load(env_file)
    ls = get_client(settings)

    if project:
        backends = list(ls.ml.list(project=project))
    else:
        backends = []
        for p in ls.projects.list():
            backends.extend(ls.ml.list(project=p.id))

    if not backends:
        info("No ML backends attached.")
        return

    rows = [
        (b.id, b.project, b.title or "", b.url or "", "interactive" if b.is_interactive else "batch")
        for b in backends
    ]
    print_table("ML Backends", ["ID", "Project", "Title", "URL", "Mode"], rows)


@app.command()
def switch(
    project_id: int = typer.Argument(help="Project ID"),
    backend: BackendChoice = typer.Argument(help="Which backend to attach (yolo, sam3, none)"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Replace the project's ML backend with the chosen one (idempotent)."""
    from ls_cli.client import get_client
    from ls_cli.config import Settings

    settings = Settings.load(env_file)
    ls = get_client(settings)
    ensure_single_backend(ls, settings, project_id, backend)
