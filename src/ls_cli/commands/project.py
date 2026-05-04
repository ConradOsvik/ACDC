"""Project management commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ls_cli.utils.backends import BackendChoice, ensure_single_backend
from ls_cli.utils.output import info, print_table, success

app = typer.Typer(help="Manage Label Studio projects.")


DEFAULT_LABEL_CONFIG = """<View>
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


@app.command()
def create(
    title: str = typer.Option("Corrosion Annotation", help="Project title"),
    label_config: Optional[Path] = typer.Option(None, help="Path to label config XML file"),
    backend: BackendChoice = typer.Option(BackendChoice.yolo, help="ML backend to attach (yolo, sam3, none)"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Idempotently create a project and attach a single ML backend."""
    from ls_cli.client import get_client
    from ls_cli.config import Settings

    settings = Settings.load(env_file)
    ls = get_client(settings)

    existing = next((p for p in ls.projects.list() if p.title == title), None)
    if existing:
        project = existing
        info(f"Project '{title}' already exists (id={project.id}), reusing")
    else:
        config_xml = label_config.read_text() if label_config else DEFAULT_LABEL_CONFIG
        project = ls.projects.create(title=title, label_config=config_xml)
        success(f"Created project '{title}' (id={project.id})")

    ensure_single_backend(ls, settings, project.id, backend)


@app.command("list")
def list_projects(
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """List all projects."""
    from ls_cli.client import get_client
    from ls_cli.config import Settings

    settings = Settings.load(env_file)
    ls = get_client(settings)
    projects = list(ls.projects.list())
    if not projects:
        info("No projects found.")
        return
    rows = [(p.id, p.title, p.task_number or 0) for p in projects]
    print_table("Projects", ["ID", "Title", "Tasks"], rows)


@app.command()
def delete(
    project_id: int = typer.Argument(help="Project ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Delete a project."""
    from ls_cli.client import get_client
    from ls_cli.config import Settings

    settings = Settings.load(env_file)
    ls = get_client(settings)
    project = ls.projects.get(id=project_id)
    if not force:
        typer.confirm(f"Delete project '{project.title}' (id={project_id})?", abort=True)
    ls.projects.delete(id=project_id)
    success(f"Deleted project '{project.title}' (id={project_id})")
