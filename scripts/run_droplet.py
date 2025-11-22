#!/usr/bin/env python
"""Prepare and simulate a single spherical water droplet.

Run from a source checkout:
    python scripts/run_droplet.py --radius 3.0 --equil-ns 1 --prod-ns 5

Key features:
- Builds droplet via prepare_water_droplet, then runs Langevin dynamics (300 K, 1 ps^-1, 1 fs).
- Separate reporting intervals for trajectories (DCD) and energies (StateDataReporter).
- Dry-run mode writes PDB/XML and prints resolved parameters without integrating.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

from openmm import LangevinIntegrator, Platform, XmlSerializer, unit
from openmm.app import DCDReporter, PDBFile, Simulation, StateDataReporter

try:  # Allow running from a source checkout without installation
    from openmm_spherical_boundaries.droplet import prepare_water_droplet
except ImportError:  # pragma: no cover - fallback for editable runs
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from openmm_spherical_boundaries.droplet import prepare_water_droplet

LOGGER = logging.getLogger("run_droplet")

DEFAULT_TRAJ_INTERVAL_PS = 10.0
DEFAULT_ENERGY_INTERVAL_PS = 1.0
DEFAULT_TEMPERATURE_K = 300.0
DEFAULT_FRICTION_PS = 1.0
DEFAULT_TIMESTEP_FS = 1.0
DEFAULT_EQUIL_NS = 1.0
DEFAULT_PROD_NS = 5.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Argument parser for a single droplet run."""
    parser = argparse.ArgumentParser(
        description="Build and run a single water droplet simulation.",
    )
    parser.add_argument("--radius", type=float, required=True, help="Droplet radius in nanometers")
    parser.add_argument("--output-pdb", default="droplet.pdb", help="Output PDB file")
    parser.add_argument("--system-xml", default="droplet.xml", help="Serialized System XML output")
    parser.add_argument(
        "--forcefield",
        nargs="+",
        default=("amber99sb.xml", "spce.xml"),
        help="Force field XML files (currently amber99sb + spce supported)",
    )
    parser.add_argument("--solvent-model", default="spce", help="Water model for Modeller.addSolvent")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.3,
        help="Padding (nm) added to the target radius before trimming",
    )
    parser.add_argument(
        "--shell-thickness",
        type=float,
        default=0.3,
        help="Thickness (nm) of the boundary shell used for elastic restraints",
    )
    parser.add_argument(
        "--shell-cutoff",
        type=float,
        default=0.6,
        help="Distance cutoff (nm) for boundary elastic bonds",
    )
    parser.add_argument(
        "--force-constant",
        type=float,
        default=None,
        help="Elastic bond force constant in kJ/mol/nm^2 (defaults: molten=1000, triangular=3000)",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=["molten", "triangular"],
        default="molten",
        help="Elastic boundary construction to use",
    )
    parser.add_argument(
        "--extra-space",
        type=float,
        default=0.15,
        help="Additional radial spacing (nm) for triangular boundary placement",
    )
    parser.add_argument(
        "--num-subdivisions",
        type=int,
        default=3,
        help="Icosahedron subdivision count for triangular boundary",
    )

    parser.add_argument(
        "--equil-ns",
        type=float,
        default=DEFAULT_EQUIL_NS,
        help="Equilibration duration (nanoseconds)",
    )
    parser.add_argument(
        "--prod-ns",
        type=float,
        default=DEFAULT_PROD_NS,
        help="Production duration (nanoseconds)",
    )
    parser.add_argument(
        "--traj-interval-ps",
        type=float,
        default=DEFAULT_TRAJ_INTERVAL_PS,
        help="Trajectory reporting interval in picoseconds",
    )
    parser.add_argument(
        "--energy-interval-ps",
        type=float,
        default=DEFAULT_ENERGY_INTERVAL_PS,
        help="Energy/StateData reporting interval in picoseconds",
    )
    parser.add_argument(
        "--traj-format",
        choices=["dcd"],
        default="dcd",
        help="Trajectory format (currently only dcd)",
    )
    parser.add_argument("--equil-traj", default="equil.dcd", help="Equilibration trajectory output")
    parser.add_argument("--prod-traj", default="prod.dcd", help="Production trajectory output")
    parser.add_argument("--equil-log", default="equil.log", help="Equilibration energy log output")
    parser.add_argument("--prod-log", default="prod.log", help="Production energy log output")
    parser.add_argument("--state-xml", default="state.xml", help="Final state XML output")
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Run an energy minimization before equilibration",
    )
    parser.add_argument(
        "--minimize-iterations",
        type=int,
        default=500,
        help="Maximum iterations for energy minimization",
    )
    parser.add_argument(
        "--warmup-start-k",
        type=float,
        default=10.0,
        help="Starting temperature (K) for linear warmup during equilibration (default: 10 K)",
    )
    parser.add_argument(
        "--warmup-duration-ns",
        type=float,
        default=0.1,
        help="Duration of temperature warmup (ns); default 0.1 ns (100 ps)",
    )
    parser.add_argument("--warmup-traj", default="warmup.dcd", help="Warmup trajectory output")
    parser.add_argument("--warmup-log", default="warmup.log", help="Warmup energy log output")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE_K,
        help="Langevin thermostat temperature in Kelvin",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=DEFAULT_FRICTION_PS,
        help="Langevin friction coefficient in 1/ps",
    )
    parser.add_argument(
        "--timestep-fs",
        type=float,
        default=DEFAULT_TIMESTEP_FS,
        help="Integrator timestep in femtoseconds",
    )
    parser.add_argument("--platform", default=None, help="OpenMM platform name (optional)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for velocity initialization and integrator (default: random)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the droplet files and print parameters without running dynamics",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-vv for debug)",
    )
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="Disable StateDataReporter output to stdout",
    )
    return parser.parse_args(argv)


def configure_logging(verbosity: int) -> None:
    """Set global logging level based on -v count."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def ns_to_steps(duration_ns: float, timestep_fs: float) -> int:
    """Convert a duration in ns to an integer step count."""
    if duration_ns <= 0:
        raise ValueError("Simulation duration must be positive (ns).")
    if timestep_fs <= 0:
        raise ValueError("Timestep must be positive (fs).")
    total_ps = duration_ns * 1000.0
    steps = int(round(total_ps / (timestep_fs * 0.001)))
    if steps <= 0:
        raise ValueError("Computed zero steps; check timestep and duration.")
    return steps


def ps_to_steps(interval_ps: float, timestep_fs: float) -> int:
    """Convert a report interval in ps to an integer step count."""
    if interval_ps <= 0:
        raise ValueError("Report interval must be positive (ps).")
    steps = int(max(1, round(interval_ps / (timestep_fs * 0.001))))
    return steps


def build_simulation(
    pdb_path: Path,
    system_xml: Path,
    temperature_k: float,
    friction_ps: float,
    timestep_fs: float,
    platform_name: str | None,
    seed: int,
) -> Simulation:
    """Instantiate a Simulation with Langevin integrator and initialized velocities."""
    LOGGER.info("Loading positions from %s", pdb_path)
    pdb = PDBFile(str(pdb_path))

    LOGGER.info("Deserializing system from %s", system_xml)
    with system_xml.open() as handle:
        system = XmlSerializer.deserialize(handle.read())

    integrator = LangevinIntegrator(
        temperature_k * unit.kelvin,
        friction_ps / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(seed)

    platform = Platform.getPlatformByName(platform_name) if platform_name else None

    simulation = Simulation(
        pdb.topology,
        system,
        integrator,
        platform=platform,
    )
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)
    return simulation


def run_phase(
    simulation: Simulation,
    steps: int,
    traj_interval_steps: int,
    energy_interval_steps: int,
    traj_path: Path,
    log_path: Path,
    log_stdout: bool,
    phase_name: str,
    warmup: tuple[float, float, int] | None = None,
) -> None:
    """Run a single phase (equil or prod) with separate traj/energy reporters."""
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    simulation.reporters = []
    simulation.reporters.append(DCDReporter(str(traj_path), traj_interval_steps))
    simulation.reporters.append(
        StateDataReporter(
            file=str(log_path),
            reportInterval=energy_interval_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            speed=True,
        )
    )
    if log_stdout:
        simulation.reporters.append(
            StateDataReporter(
                file=sys.stdout,
                reportInterval=energy_interval_steps,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                speed=True,
            )
        )

    LOGGER.info(
        "Starting %s phase for %d steps (traj every %d steps, energy every %d steps)",
        phase_name,
        steps,
        traj_interval_steps,
        energy_interval_steps,
    )
    if warmup:
        start_k, end_k, warmup_steps = warmup
        done = 0
        # Use smaller chunks so we can adjust temperature smoothly.
        chunk = max(1, min(traj_interval_steps, energy_interval_steps, 1000))
        while done < steps:
            if done < warmup_steps:
                frac = min(1.0, done / max(1, warmup_steps))
                target_temp = start_k + frac * (end_k - start_k)
                simulation.integrator.setTemperature(target_temp * unit.kelvin)
            step_now = min(chunk, steps - done)
            simulation.step(step_now)
            done += step_now
        # Ensure final temperature at end of warmup.
        if warmup_steps > 0:
            simulation.integrator.setTemperature(end_k * unit.kelvin)
    else:
        simulation.step(steps)
    LOGGER.info("%s phase complete", phase_name.capitalize())


def write_final_state(simulation: Simulation, path: Path) -> None:
    """Serialize the final state (positions + velocities) to XML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    with path.open("w") as handle:
        handle.write(XmlSerializer.serialize(state))
    LOGGER.info("Wrote final state to %s", path)


def summarize_parameters(
    args: argparse.Namespace,
    seed: int,
    equil_steps: int,
    prod_steps: int,
    traj_steps: int,
    energy_steps: int,
    warmup_steps: int | None,
) -> None:
    """Log the resolved simulation parameters and outputs."""
    LOGGER.info(
        "Simulation parameters: radius=%.3f nm, temperature=%.1f K, friction=%.3f 1/ps, "
        "timestep=%.3f fs, equil=%.3f ns (%d steps), prod=%.3f ns (%d steps), "
        "traj interval=%.1f ps (%d steps), energy interval=%.1f ps (%d steps), seed=%d, platform=%s",
        args.radius,
        args.temperature,
        args.friction,
        args.timestep_fs,
        args.equil_ns,
        equil_steps,
        args.prod_ns,
        prod_steps,
        args.traj_interval_ps,
        traj_steps,
        args.energy_interval_ps,
        energy_steps,
        seed,
        args.platform or "default",
    )
    if warmup_steps:
        LOGGER.info(
            "Warmup: start %.1f K -> target %.1f K over %d steps",
            args.warmup_start_k,
            args.temperature,
            warmup_steps,
        )
    LOGGER.info(
        "Outputs: PDB=%s, system XML=%s, equil traj=%s, prod traj=%s, equil log=%s, prod log=%s, final state=%s",
        args.output_pdb,
        args.system_xml,
        args.equil_traj,
        args.prod_traj,
        args.equil_log,
        args.prod_log,
        args.state_xml,
    )
    if warmup_steps:
        LOGGER.info(
            "Warmup outputs: traj=%s, log=%s",
            args.warmup_traj,
            args.warmup_log,
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    seed = args.seed if args.seed is not None else random.randrange(1, 2**31 - 1)

    prepare_water_droplet(
        radius=args.radius,
        output_pdb=args.output_pdb,
        system_xml=args.system_xml,
        forcefield_files=args.forcefield,
        solvent_model=args.solvent_model,
        padding=args.padding,
        shell_thickness=args.shell_thickness,
        shell_cutoff=args.shell_cutoff,
        force_constant=args.force_constant,
        boundary_mode=args.boundary_mode,
        extra_space=args.extra_space,
        num_subdivisions=args.num_subdivisions,
    )

    equil_steps = ns_to_steps(args.equil_ns, args.timestep_fs)
    prod_steps = ns_to_steps(args.prod_ns, args.timestep_fs)
    traj_steps = ps_to_steps(args.traj_interval_ps, args.timestep_fs)
    energy_steps = ps_to_steps(args.energy_interval_ps, args.timestep_fs)
    warmup_steps = None
    warmup_ns = args.warmup_duration_ns if args.warmup_start_k is not None else 0.0
    if args.warmup_start_k is not None:
        warmup_steps = min(equil_steps, ns_to_steps(warmup_ns, args.timestep_fs))
    summarize_parameters(args, seed, equil_steps, prod_steps, traj_steps, energy_steps, warmup_steps)

    if args.dry_run:
        LOGGER.info("Dry-run complete; skipping dynamics.")
        return

    simulation = build_simulation(
        pdb_path=Path(args.output_pdb),
        system_xml=Path(args.system_xml),
        temperature_k=args.temperature,
        friction_ps=args.friction,
        timestep_fs=args.timestep_fs,
        platform_name=args.platform,
        seed=seed,
    )

    if args.minimize:
        LOGGER.info("Running energy minimization (max %d iterations)", args.minimize_iterations)
        simulation.minimizeEnergy(maxIterations=args.minimize_iterations)
        LOGGER.info("Minimization complete")

    # Optional warmup before equilibration
    if warmup_steps:
        run_phase(
            simulation=simulation,
            steps=warmup_steps,
            traj_interval_steps=min(traj_steps, warmup_steps),
            energy_interval_steps=min(energy_steps, warmup_steps),
            traj_path=Path(args.warmup_traj),
            log_path=Path(args.warmup_log),
            log_stdout=not args.no_stdout,
            phase_name="warmup",
            warmup=(args.warmup_start_k, args.temperature, warmup_steps),
        )

    run_phase(
        simulation=simulation,
        steps=equil_steps,
        traj_interval_steps=min(traj_steps, equil_steps),
        energy_interval_steps=min(energy_steps, equil_steps),
        traj_path=Path(args.equil_traj),
        log_path=Path(args.equil_log),
        log_stdout=not args.no_stdout,
        phase_name="equilibration",
    )
    run_phase(
        simulation=simulation,
        steps=prod_steps,
        traj_interval_steps=min(traj_steps, prod_steps),
        energy_interval_steps=min(energy_steps, prod_steps),
        traj_path=Path(args.prod_traj),
        log_path=Path(args.prod_log),
        log_stdout=not args.no_stdout,
        phase_name="production",
    )
    write_final_state(simulation, Path(args.state_xml))


if __name__ == "__main__":  # pragma: no cover
    main()
