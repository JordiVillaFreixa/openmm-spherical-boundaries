"""Command-line interface for the spherical boundary utilities."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .droplet import prepare_water_droplet
from .prepare_md_system import prepare_md_system
from .prepare_ref_sim import prepare_reference_system
from .simulation import run_serialized_simulation
from .triangular_model import prepare_triangular_boundary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tools for building OpenMM spherical boundary systems",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use -vv for debug)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _setup_prepare_md_parser(subparsers)
    _setup_prepare_triangular_parser(subparsers)
    _setup_prepare_reference_parser(subparsers)
    _setup_prepare_droplet_parser(subparsers)
    _setup_run_parser(subparsers)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    handler = getattr(args, "handler")
    handler(args)


def _setup_prepare_md_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare-md",
        help="Trim a solvated box and build an elastic network boundary",
    )
    parser.set_defaults(handler=_handle_prepare_md)
    parser.add_argument("input_pdb", help="Input PDB structure")
    parser.add_argument("--output-pdb", default="multiresPrep.pdb")
    parser.add_argument("--system-xml", default="system.xml")
    parser.add_argument(
        "--forcefield",
        nargs="+",
        default=("amber99sb.xml", "spce.xml"),
        help="Force field XML definitions",
    )
    parser.add_argument(
        "--shell-center",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(4.0, 4.0, 4.0),
        help="Center of the spherical region (nm)",
    )
    parser.add_argument("--shell-radius", type=float, default=2.0, help="Radius of inner shell (nm)")
    parser.add_argument(
        "--shell-thickness",
        type=float,
        default=1.0,
        help="Thickness of ENM shell (nm)",
    )
    parser.add_argument(
        "--shell-cutoff",
        type=float,
        default=0.6,
        help="Distance cutoff for ENM bonds (nm)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display a histogram of bond lengths",
    )


def _setup_prepare_triangular_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare-triangular",
        help="Generate a triangular water boundary shell",
    )
    parser.set_defaults(handler=_handle_prepare_triangular)
    parser.add_argument("input_pdb", help="Input PDB structure")
    parser.add_argument("--output-pdb", default="multiresTriag.pdb")
    parser.add_argument("--system-xml", default="system.xml")
    parser.add_argument(
        "--forcefield",
        nargs="+",
        default=("amber99sb.xml", "spce.xml"),
        help="Force field XML definitions",
    )
    parser.add_argument(
        "--shell-center",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(4.0, 4.0, 4.0),
    )
    parser.add_argument("--shell-radius", type=float, default=2.0)
    parser.add_argument("--shell-cutoff", type=float, default=0.4)
    parser.add_argument("--extra-space", type=float, default=0.15)
    parser.add_argument("--num-subdivisions", type=int, default=3)
    parser.add_argument("--plot", action="store_true")


def _setup_prepare_reference_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare-reference",
        help="Write an unmodified periodic reference system",
    )
    parser.set_defaults(handler=_handle_prepare_reference)
    parser.add_argument("input_pdb", help="Input PDB file")
    parser.add_argument("--output-pdb", default="multiresRef.pdb")
    parser.add_argument("--system-xml", default="system_ref.xml")
    parser.add_argument(
        "--forcefield",
        nargs="+",
        default=("amber99sb.xml", "spce.xml"),
    )


def _setup_prepare_droplet_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare-droplet",
        help="Create a standalone spherical water droplet",
    )
    parser.set_defaults(handler=_handle_prepare_droplet)
    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Droplet radius in nanometers",
    )
    parser.add_argument("--output-pdb", default="droplet.pdb")
    parser.add_argument("--system-xml", default="droplet.xml")
    parser.add_argument(
        "--forcefield",
        nargs="+",
        default=("amber99sb.xml", "spce.xml"),
        help="Force field XML definitions (currently only amber99sb+spce supported)",
    )
    parser.add_argument(
        "--solvent-model",
        default="spce",
        help="Water model passed to Modeller.addSolvent (default: spce)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.3,
        help="Extra padding (nm) added around the requested radius before trimming",
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
        help="Elastic bond force constant in kJ/mol/nm^2 (defaults depend on mode)",
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


def _setup_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Run a simple simulation using serialized inputs",
    )
    parser.set_defaults(handler=_handle_run)
    parser.add_argument("pdb", help="Input PDB file")
    parser.add_argument("system_xml", help="Serialized system (XML)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--report-interval", type=int, default=10)
    parser.add_argument("--traj-file", default="trajectory.dcd")
    parser.add_argument(
        "--traj-format",
        choices=["dcd", "pdb"],
        default="dcd",
        help="Trajectory format to write (default: dcd)",
    )
    parser.add_argument("--platform", default=None, help="OpenMM platform (optional)")
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="Do not print StateDataReporter output to stdout",
    )


def _handle_prepare_md(args: argparse.Namespace) -> None:
    prepare_md_system(
        input_pdb=args.input_pdb,
        output_pdb=args.output_pdb,
        system_xml=args.system_xml,
        forcefield_files=args.forcefield,
        shell_center=args.shell_center,
        shell_radius=args.shell_radius,
        shell_thickness=args.shell_thickness,
        shell_cutoff=args.shell_cutoff,
        plot_histogram=args.plot,
    )


def _handle_prepare_triangular(args: argparse.Namespace) -> None:
    prepare_triangular_boundary(
        input_pdb=args.input_pdb,
        output_pdb=args.output_pdb,
        system_xml=args.system_xml,
        forcefield_files=args.forcefield,
        shell_center=args.shell_center,
        shell_radius=args.shell_radius,
        shell_cutoff=args.shell_cutoff,
        extra_space=args.extra_space,
        num_subdivisions=args.num_subdivisions,
        plot_histogram=args.plot,
    )


def _handle_prepare_reference(args: argparse.Namespace) -> None:
    prepare_reference_system(
        input_pdb=args.input_pdb,
        output_pdb=args.output_pdb,
        system_xml=args.system_xml,
        forcefield_files=args.forcefield,
    )


def _handle_prepare_droplet(args: argparse.Namespace) -> None:
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


def _handle_run(args: argparse.Namespace) -> None:
    run_serialized_simulation(
        pdb_path=args.pdb,
        system_xml=args.system_xml,
        steps=args.steps,
        report_interval=args.report_interval,
        traj_path=args.traj_file,
        trajectory_format=args.traj_format,
        platform_name=args.platform,
        log_stdout=not args.no_stdout,
    )


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":  # pragma: no cover
    main()
