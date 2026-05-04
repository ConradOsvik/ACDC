"""Rich console output helpers."""

from __future__ import annotations

from typing import Any, Sequence

from rich.console import Console
from rich.table import Table

console = Console()


def print_table(title: str, columns: list[str], rows: Sequence[Sequence[Any]]) -> None:
    table = Table(title=title)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*(str(v) for v in row))
    console.print(table)


def success(msg: str) -> None:
    console.print(f"[green]{msg}[/green]")


def error(msg: str) -> None:
    console.print(f"[red]{msg}[/red]")


def info(msg: str) -> None:
    console.print(msg)
