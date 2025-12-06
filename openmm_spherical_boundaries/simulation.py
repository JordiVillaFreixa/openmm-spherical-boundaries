"""Utility routine for running serialized OpenMM systems."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from openmm import CustomIntegrator, Platform, XmlSerializer
from openmm.app import DCDReporter, PDBFile, Simulation, StateDataReporter

LOGGER = logging.getLogger(__name__)


def run_serialized_simulation(
    pdb_path: Path | str,
    system_xml: Path | str,
    steps: int = 1000,
    traj_path: Path | str = "trajectory.dcd",
    report_interval: int = 10,
    platform_name: Optional[str] = None,
    log_stdout: bool = True,
    trajectory_format: str = "dcd",
) -> None:
    """Run a short simulation using a pre-built system/pdb pair."""

    pdb_file = Path(pdb_path)
    xml_file = Path(system_xml)
    LOGGER.info("Loading positions from %s", pdb_file)
    pdb = PDBFile(str(pdb_file))

    LOGGER.info("Deserializing system from %s", xml_file)
    with xml_file.open() as handle:
        system = XmlSerializer.deserialize(handle.read())

    integrator = CustomIntegrator(0.001)
    integrator.addPerDofVariable("x1", 0)
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
    integrator.addConstrainVelocities()

    platform = None
    if platform_name:
        platform = Platform.getPlatformByName(platform_name)

    simulation = Simulation(
        pdb.topology,
        system,
        integrator,
        platform=platform,
    )
    simulation.context.setPositions(pdb.positions)

    traj_file = Path(traj_path)
    traj_file.parent.mkdir(parents=True, exist_ok=True)
    reporter = _create_trajectory_reporter(trajectory_format, traj_file, report_interval)
    simulation.reporters.append(reporter)

    if log_stdout:
        simulation.reporters.append(
            StateDataReporter(
                file=sys.stdout,
                reportInterval=report_interval,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                speed=True,
            )
        )

    LOGGER.info("Starting simulation for %d steps", steps)
    simulation.step(steps)
    LOGGER.info("Simulation complete")


def _create_trajectory_reporter(
    fmt: str,
    path: Path,
    report_interval: int,
):
    fmt = fmt.lower()
    if fmt == "dcd":
        return DCDReporter(str(path), report_interval)
    if fmt == "pdb":
        from openmm.app import PDBReporter

        return PDBReporter(str(path), report_interval)
    raise ValueError(f"Unsupported trajectory format '{fmt}'. Expected 'dcd' or 'pdb'.")
