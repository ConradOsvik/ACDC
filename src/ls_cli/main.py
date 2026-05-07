"""ls-cli — Label Studio project and ML backend management."""

from __future__ import annotations

import typer

from ls_cli.commands import backend, compare, project, sam3, stack, yolo

app = typer.Typer(
    name="ls-cli",
    help="CLI for managing Label Studio projects and ML backends.",
    no_args_is_help=True,
)

app.command("init")(stack.init)
app.command("up")(stack.up)
app.command("down")(stack.down)
app.add_typer(project.app, name="project")
app.add_typer(backend.app, name="backend")
app.add_typer(sam3.app, name="sam3")
app.add_typer(yolo.app, name="yolo")
app.add_typer(compare.app, name="compare")


if __name__ == "__main__":
    app()
