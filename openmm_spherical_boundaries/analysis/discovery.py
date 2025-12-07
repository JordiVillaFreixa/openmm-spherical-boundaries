"""Helpers for locating droplet jobs on disk."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .metrics import SimulationDataset

EXPECTED_OUTPUTS = {
    "topology": "droplet.pdb",
    "trajectories": ["warmup.dcd", "equil.dcd", "prod.dcd"],
}


def discover_simulations(
    root: str | Path,
    *,
    include_variants: Iterable[str] | None = None,
) -> dict[str, dict[str, SimulationDataset]]:
    """Return SimulationDataset entries discovered under a setup_droplet_jobs folder."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Simulation root does not exist: {root_path}")

    filters = set(include_variants or [])
    layout: dict[str, dict[str, SimulationDataset]] = {}

    for variant_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        if variant_dir.name == "script":
            continue
        if filters and variant_dir.name not in filters:
            continue

        replicas: dict[str, SimulationDataset] = {}
        for replica_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            params_path = replica_dir / "params.json"
            if not params_path.exists():
                continue
            with params_path.open() as handle:
                payload = json.load(handle)

            metadata = payload.get("params", {})
            metadata["command"] = payload.get("command")
            metadata["replica_index"] = payload.get("replica_index")
            metadata["replica_label"] = payload.get("replica_label", replica_dir.name)

            topology_path = replica_dir / EXPECTED_OUTPUTS["topology"]
            trajectory_paths = [replica_dir / name for name in EXPECTED_OUTPUTS["trajectories"]]
            missing_outputs = [path for path in [topology_path, *trajectory_paths] if not path.exists()]
            if missing_outputs:
                # Skip replicas that have not produced the expected files yet.
                continue

            dataset = SimulationDataset(
                label=f"{variant_dir.name}/{metadata['replica_label']}",
                topology_path=topology_path,
                trajectory_paths=trajectory_paths,
                metadata=metadata,
            )
            replicas[metadata["replica_label"]] = dataset

        if replicas:
            layout[variant_dir.name] = replicas

    return layout
