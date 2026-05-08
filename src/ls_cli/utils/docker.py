"""Docker / docker-compose subprocess helpers."""

from __future__ import annotations

import subprocess
import sys


def docker_cp(src: str, dest: str) -> None:
    _run(["docker", "cp", src, dest])


def docker_exec(container: str, *cmd: str) -> None:
    _run(["docker", "exec", container, *cmd])


def docker_exec_python(container: str, script: str) -> str:
    """Run a Python script string inside the container via stdin, return stdout."""
    result = subprocess.run(
        ["docker", "exec", "-i", container, "python", "-"],
        input=script,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise SystemExit(f"Script in container failed:\n{result.stderr}")
    return result.stdout


def compose_up(
    *compose_files: str,
    services: list[str] | None = None,
    env_file: str | None = None,
    recreate: bool = False,
    wait: bool = False,
    build: bool = False,
) -> None:
    cmd = _compose_cmd(compose_files, env_file)
    cmd += ["up", "-d"]
    if build:
        cmd.append("--build")
    if recreate:
        cmd.append("--force-recreate")
    if wait:
        cmd.append("--wait")
    if services:
        cmd += services
    _run(cmd)


def compose_down(*compose_files: str, env_file: str | None = None) -> None:
    cmd = _compose_cmd(compose_files, env_file)
    cmd.append("down")
    _run(cmd)


def compose_restart(*compose_files: str, service: str, env_file: str | None = None) -> None:
    cmd = _compose_cmd(compose_files, env_file)
    cmd += ["restart", service]
    _run(cmd)


def container_inspect_env(container: str, var: str) -> str | None:
    result = subprocess.run(
        ["docker", "exec", container, "printenv", var],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def container_host_port(container: str, internal_port: int) -> str | None:
    """Return the host port number mapped to internal_port, or None if not found."""
    result = subprocess.run(
        ["docker", "port", container, str(internal_port)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        # Lines look like "0.0.0.0:9091" or "[::]:9091"
        host_port = line.strip().rsplit(":", 1)[-1]
        if host_port.isdigit():
            return host_port
    return None


def resolve_backend_url(configured_url: str, container: str, internal_port: int = 9090) -> str:
    """Return a reachable URL for the backend.

    Tries the configured URL first (quick health check). If that fails — e.g.
    the hostname is a Docker-internal service name not reachable from the host —
    falls back to querying `docker port` and building a localhost URL.
    """
    import urllib.request

    url = configured_url.rstrip("/")
    try:
        urllib.request.urlopen(f"{url}/health", timeout=2)
        return url
    except Exception:
        pass

    host_port = container_host_port(container, internal_port)
    if host_port:
        return f"http://localhost:{host_port}"
    return url


def container_is_running(container: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _compose_cmd(compose_files: tuple[str, ...], env_file: str | None) -> list[str]:
    cmd = ["docker", "compose"]
    if env_file:
        cmd += ["--env-file", env_file]
    for f in compose_files:
        cmd += ["-f", f]
    return cmd


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(1)
    return result
