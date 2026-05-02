"""YOLO backend: deploy, status, export, visualize, train, metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ls_cli.utils.docker import (
    compose_restart,
    container_inspect_env,
    container_is_running,
    docker_cp,
    docker_exec,
)
from ls_cli.utils.output import console, error, info, success

app = typer.Typer(help="Manage the YOLO segmentation backend.")

CONTAINER_NAME = "yolo-seg"
WEIGHTS_DIR = "/data/weights"


@app.command()
def deploy(
    weights: Path = typer.Argument(help="Path to .pt weights file"),
    container: str = typer.Option(CONTAINER_NAME, help="Docker container name"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Deploy YOLO weights: copy into the volume and restart so the model picks them up."""
    from ls_cli.config import Settings

    settings = Settings.load(env_file, require_api_key=False)

    if not weights.is_file():
        error(f"Weights file not found: {weights}")
        raise typer.Exit(1)
    if weights.suffix != ".pt":
        error(f"Expected a .pt file, got: {weights.name}")
        raise typer.Exit(1)
    if not container_is_running(container):
        error(f"Container '{container}' is not running. Start it first (ls-cli up).")
        raise typer.Exit(1)

    dest = f"{WEIGHTS_DIR}/{weights.name}"
    info(f"Copying {weights} → {container}:{dest}")
    docker_exec(container, "mkdir", "-p", WEIGHTS_DIR)
    docker_cp(str(weights), f"{container}:{dest}")

    if not settings.compose_models_file:
        error("Could not locate the models compose file. Restart the container manually.")
        raise typer.Exit(1)

    info(f"Restarting {container}...")
    compose_restart(settings.compose_models_file, service=container, env_file=settings.env_file)
    success(f"Deployed {weights.name}. The backend will load it on the next prediction.")


@app.command()
def status(
    container: str = typer.Option(CONTAINER_NAME, help="Docker container name"),
) -> None:
    """Show the YOLO backend's current state."""
    running = container_is_running(container)
    info(f"Container:  {container}")
    info(f"Running:    {'yes' if running else 'no'}")
    if running:
        weights_dir = container_inspect_env(container, "YOLO_WEIGHTS_DIR") or WEIGHTS_DIR
        conf = container_inspect_env(container, "YOLO_CONF")
        info(f"Weights:    {weights_dir} (newest .pt loaded automatically)")
        info(f"Confidence: {conf or 'unknown'}")


@app.command()
def export(
    project_id: Optional[int] = typer.Option(None, help="Project ID (auto-resolved if only one)"),
    output_dir: Path = typer.Option(Path("exports/yolo_dataset"), help="Where to write the dataset"),
    split: float = typer.Option(0.8, help="Train fraction (rest goes to val)"),
    seed: int = typer.Option(42, help="Random seed for the split"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Export annotated tasks as a YOLO segmentation dataset."""
    from ls_cli.client import get_client
    from ls_cli.config import Settings
    from ls_cli.utils.dataset import export_yolo_dataset

    settings = Settings.load(env_file)
    ls = get_client(settings)
    pid = _resolve_project_id(ls, project_id)
    project = ls.projects.get(id=pid)
    info(f"Project: {project.title} (id={pid})")

    classes, exported, skipped = export_yolo_dataset(
        settings, ls, pid, output_dir, split=split, seed=seed,
    )
    info(f"Classes: {classes}")
    success(f"Exported {exported} tasks ({skipped} skipped) → {output_dir.resolve()}")


@app.command()
def visualize(
    dataset_dir: Path = typer.Option(Path("exports/yolo_dataset"), help="Dataset directory"),
    split: str = typer.Option("all", help="Which split to render (train, val, all)"),
    max_images: Optional[int] = typer.Option(None, "--max-images", "-n", help="Stop after N images"),
    save_dir: Optional[Path] = typer.Option(None, help="Save PNGs here instead of showing interactively"),
) -> None:
    """Render image + polygon overlays for an exported dataset."""
    from ls_cli.utils.dataset import visualize_yolo_dataset

    if split not in ("train", "val", "all"):
        error(f"--split must be one of: train, val, all (got {split})")
        raise typer.Exit(1)

    shown, issues = visualize_yolo_dataset(
        dataset_dir, split=split, max_images=max_images, save_dir=save_dir,
    )
    info(f"Visualised {shown} image(s).")
    if issues:
        info(f"\n{len(issues)} issue(s):")
        for msg in issues:
            info(f"  [!] {msg}")


@app.command()
def train(
    project_id: Optional[int] = typer.Option(None, help="Project ID (auto-resolved if only one)"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
) -> None:
    """Trigger training on the YOLO backend.

    Equivalent to clicking 'Start Training' in the Label Studio UI: LS sends a
    START_TRAINING webhook to the backend, which exports the dataset, trains,
    saves new weights, and reloads itself in-process.
    """
    from ls_cli.client import get_client
    from ls_cli.config import Settings

    settings = Settings.load(env_file)
    ls = get_client(settings)
    pid = _resolve_project_id(ls, project_id)

    backends = [b for b in ls.ml.list(project=pid) if (b.title or "").startswith("YOLO")]
    if not backends:
        error(f"No YOLO backend attached to project {pid}. Run: ls-cli backend switch {pid} yolo")
        raise typer.Exit(1)

    backend = backends[0]
    info(f"Triggering training on '{backend.title}' (id={backend.id}) for project {pid}...")
    _start_training(settings, ls, backend.id)
    success("Training started. Check the YOLO container logs for progress.")


def _start_training(settings, ls, backend_id: int) -> None:
    """Start a training run via the LS API. Tries the SDK first, then a raw POST."""
    import requests

    from ls_cli.client import auth_headers

    train_method = getattr(ls.ml, "train", None)
    if callable(train_method):
        try:
            train_method(id=backend_id)
            return
        except Exception:
            pass

    base = settings.label_studio_url.rstrip("/")
    resp = requests.post(
        f"{base}/api/ml/{backend_id}/train/",
        headers=auth_headers(ls),
        timeout=30,
    )
    resp.raise_for_status()


@app.command()
def runs(
    sort_by: str = typer.Option(
        "mtime", help="Sort key: mtime | mAP50 | mAP50-95 | epochs"
    ),
) -> None:
    """List all training runs with key hyperparameters and final metrics."""
    from ls_cli.utils.metrics import KEY_HPARAMS, KEY_METRICS, list_runs
    from ls_cli.utils.output import print_table

    all_runs = list_runs()
    if not all_runs:
        info("No training runs found in runs/yolo/. Run training first.")
        return

    # Sort
    sort_key_map = {
        "mtime": lambda r: r.path.stat().st_mtime,
        "mAP50": lambda r: r.best_metrics.get("metrics/mAP50(M)") or r.best_metrics.get("metrics/mAP50(B)") or 0,
        "mAP50-95": lambda r: r.best_metrics.get("metrics/mAP50-95(M)") or r.best_metrics.get("metrics/mAP50-95(B)") or 0,
        "epochs": lambda r: r.epochs_completed,
    }
    keyfn = sort_key_map.get(sort_by, sort_key_map["mtime"])
    all_runs.sort(key=keyfn, reverse=True)

    columns = ["Run", "Epochs", "imgsz", "batch", "Box mAP50", "Box mAP50-95", "Mask mAP50", "Mask mAP50-95"]
    rows = []
    for r in all_runs:
        rows.append((
            r.name,
            f"{r.epochs_completed}/{int(r.hparams.get('epochs', 0)) or '?'}",
            r.hparams.get("imgsz", "—"),
            r.hparams.get("batch", "—"),
            _fmt(r.best_metrics.get("metrics/mAP50(B)")),
            _fmt(r.best_metrics.get("metrics/mAP50-95(B)")),
            _fmt(r.best_metrics.get("metrics/mAP50(M)")),
            _fmt(r.best_metrics.get("metrics/mAP50-95(M)")),
        ))
    print_table(f"YOLO training runs ({len(all_runs)} total, sorted by {sort_by})", columns, rows)


@app.command()
def metrics(
    run: Optional[str] = typer.Option(None, "--run", "-r", help="Run name (default: latest)"),
    export: Optional[Path] = typer.Option(None, "--export", help="Copy plots + CSV + args to this dir"),
) -> None:
    """Show detailed metrics for a training run, or export them for a report."""
    from ls_cli.utils.metrics import KEY_HPARAMS, export_run, f1_from_pr, list_runs, load_run
    from ls_cli.utils.output import print_table

    runs_list = list_runs()
    if not runs_list:
        error("No training runs found in runs/yolo/. Run training first.")
        raise typer.Exit(1)

    if run:
        match = next((r for r in runs_list if r.name == run), None)
        if not match:
            error(f"No run named '{run}'. Try: ls-cli yolo runs")
            raise typer.Exit(1)
        target = match
    else:
        target = runs_list[0]

    info(f"Run: {target.name}")
    info(f"Path: {target.path}")
    info("")

    # Hyperparameters
    if target.hparams:
        rows = [(k, target.hparams.get(k, "—")) for k in KEY_HPARAMS]
        print_table("Hyperparameters", ["Key", "Value"], rows)
        info("")

    # Metrics — best epoch
    if target.best_metrics:
        bp = target.best_metrics.get("metrics/precision(B)") or 0
        br = target.best_metrics.get("metrics/recall(B)") or 0
        mp = target.best_metrics.get("metrics/precision(M)") or 0
        mr = target.best_metrics.get("metrics/recall(M)") or 0
        rows = [
            ("Box mAP50",      _fmt(target.best_metrics.get("metrics/mAP50(B)"))),
            ("Box mAP50-95",   _fmt(target.best_metrics.get("metrics/mAP50-95(B)"))),
            ("Box precision",  _fmt(bp)),
            ("Box recall",     _fmt(br)),
            ("Box F1",         _fmt(f1_from_pr(bp, br))),
            ("Mask mAP50",     _fmt(target.best_metrics.get("metrics/mAP50(M)"))),
            ("Mask mAP50-95",  _fmt(target.best_metrics.get("metrics/mAP50-95(M)"))),
            ("Mask precision", _fmt(mp)),
            ("Mask recall",    _fmt(mr)),
            ("Mask F1",        _fmt(f1_from_pr(mp, mr))),
        ]
        print_table(f"Best-epoch metrics ({target.epochs_completed}/{int(target.hparams.get('epochs', 0)) or '?'} epochs)", ["Metric", "Value"], rows)
        info("")
    else:
        info("No metrics yet (results.csv missing or empty).")

    # Plot inventory
    if target.has_plots:
        info(f"Plots available in {target.path} (results.png, confusion_matrix.png, P/R/F1/PR_curve.png, val_batch*.jpg)")
    else:
        info("No plots — was the run trained with plots=True?")

    if export:
        copied = export_run(target, export)
        success(f"Exported {len(copied)} files → {export}")
        info("Per-class metrics: see val output during training (printed to docker logs).")


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "—"
    return f"{f:.4f}"


def _resolve_project_id(ls, project_id: int | None) -> int:
    if project_id is not None:
        return project_id
    projects = list(ls.projects.list())
    if len(projects) == 1:
        return projects[0].id
    if not projects:
        error("No projects found. Run: ls-cli project create")
        raise typer.Exit(1)
    error("Multiple projects exist — pass --project-id:")
    for p in projects:
        error(f"  {p.id}: {p.title}")
    raise typer.Exit(1)
