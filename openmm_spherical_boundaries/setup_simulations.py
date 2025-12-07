"""Utilities for setting up droplet simulation jobs.

Example:
    from openmm_spherical_boundaries.setup_simulations import setup_droplet_jobs
    commands = setup_droplet_jobs(
        simulation_folder="runs",
        variants=[{"radius": r} for r in (2.0, 2.5, 3.0)],
        base_params={"equil_ns": 1.0, "prod_ns": 5.0},
        replicas=3,
    )
    for cmd in commands:
        print(cmd)

This prepares per-variant folders, writes a params.json, and emits ready-to-run
commands that call ``scripts/run_droplet.py``.  Pass ``replicas`` to create multiple
subdirectories per parameter set (e.g., replica-01, replica-02).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import shlex
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

MAX_NAME_PARAMS_WARNING = 3
UNIT_SUFFIXES: dict[str, str] = {
    "radius": "nm",
    "temperature": "K",
    "friction": "perps",
    "timestep_fs": "fs",
    "equil_ns": "ns",
    "prod_ns": "ns",
}


def setup_droplet_jobs(
    simulation_folder: str | Path,
    variants: Sequence[Mapping[str, Any]],
    base_params: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    *,
    replicas: int = 1,
) -> list[str]:
    """Prepare per-variant folders and commands for droplet runs.

    Args:
        simulation_folder: Root directory containing all variant subfolders.
        variants: Sequence of mappings with the varying parameters for each job.
        base_params: Common parameters applied to every job (overridden by variants).
        overwrite: If True, existing job folders are deleted and recreated; if False,
            existing folders are reused and outputs may be overwritten in place.
        replicas: Number of replicas to create per parameter set (default: 1).

    Returns:
        List of shell command strings (one per job), ready to run.
    """

    base_params = dict(base_params or {})
    if replicas < 1:
        raise ValueError("replicas must be at least 1")
    root = Path(simulation_folder)
    root.mkdir(parents=True, exist_ok=True)

    script_dir = root / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    source_script = repo_root / "scripts" / "run_droplet.py"
    dest_script = script_dir / "run_droplet.py"
    shutil.copy2(source_script, dest_script)

    commands: list[str] = []
    for variant in variants:
        variant_params = dict(variant)
        job_params = {**base_params, **variant_params}

        name = _folder_name_from_params(variant_params)
        job_path = root / name

        if job_path.exists():
            if overwrite:
                LOGGER.info("Clearing existing job folder %s", job_path)
                shutil.rmtree(job_path)
            else:
                LOGGER.info("Reusing existing job folder %s (overwrite disabled)", job_path)
        job_path.mkdir(parents=True, exist_ok=True)

        commands.extend(
            _write_replica_jobs(
                job_path=job_path,
                job_params=job_params,
                replicas=replicas,
                script_path=dest_script,
            )
        )

    return commands


def _write_replica_jobs(
    job_path: Path,
    job_params: Mapping[str, Any],
    replicas: int,
    script_path: Path,
) -> list[str]:
    """Write params.json/commands for each replica under the provided job path."""

    commands: list[str] = []
    for replica_index in range(replicas):
        replica_label = f"replica-{replica_index + 1:02d}"
        replica_path = job_path / replica_label
        replica_path.mkdir(parents=True, exist_ok=True)

        command = _build_command(job_params, cwd=replica_path, script_path=script_path)

        payload: dict[str, Any] = {
            "params": job_params,
            "command": command,
            "replica_index": replica_index,
            "replica_label": replica_label,
        }

        params_path = replica_path / "params.json"
        with params_path.open("w") as handle:
            json.dump(payload, handle, indent=2)

        commands.append(command)

    return commands


def _folder_name_from_params(params: Mapping[str, Any]) -> str:
    """Create a deterministic folder name from varying parameters."""
    if not params:
        return "job"
    parts: list[str] = []
    for key in sorted(params.keys()):
        value = params[key]
        suffix = UNIT_SUFFIXES.get(key, "")
        parts.append(f"{key}-{_format_value_for_name(value)}{suffix}")
    if len(parts) > MAX_NAME_PARAMS_WARNING:
        LOGGER.warning(
            "Folder name includes %d parameters; consider reducing to avoid long paths.",
            len(parts),
        )
    return "_".join(parts)


def _format_value_for_name(value: Any) -> str:
    """Format a value for inclusion in a folder name (dots -> p)."""
    if isinstance(value, float):
        text = f"{value}"
    else:
        text = str(value)
    text = text.replace(".", "p")
    text = text.replace("/", "-")
    return text


def _build_command(params: Mapping[str, Any], cwd: Path, script_path: Path) -> str:
    """Return a shell command to run the droplet with the provided parameters."""
    args = []
    for key, value in params.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(_quote_sequence(value))
            continue
        args.extend([flag, shlex.quote(str(value))])

    outputs = [
        "--output-pdb droplet.pdb",
        "--system-xml droplet.xml",
        "--warmup-traj warmup.dcd",
        "--equil-traj equil.dcd",
        "--prod-traj prod.dcd",
        "--warmup-log warmup.log",
        "--equil-log equil.log",
        "--prod-log prod.log",
        "--state-xml state.xml",
    ]

    rel_script = os.path.relpath(script_path, cwd)
    joined_args = " ".join(args + outputs)
    cmd = f"python {shlex.quote(rel_script)} {joined_args}"
    return f"(cd {shlex.quote(str(cwd))} && {cmd})\n"


def _quote_sequence(seq: Iterable[Any]) -> list[str]:
    """Quote items in a sequence for shell safety."""
    return [shlex.quote(str(item)) for item in seq]


__all__ = ["setup_droplet_jobs"]
