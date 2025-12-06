"""Generate a triangular shell of water molecules for boundary conditions."""

from __future__ import annotations

import io
import logging
from math import cos, pi, sin
from pathlib import Path
from typing import List, Sequence

from openmm import HarmonicBondForce, Vec3, XmlSerializer, unit
from openmm.app import CutoffNonPeriodic, ForceField, Modeller, PDBFile

from .geometry import deduplicate_vertices, icosahedron, subdivide
from .utils import clean_openmm_positions, distance

LOGGER = logging.getLogger(__name__)


def prepare_triangular_boundary(
    input_pdb: Path | str,
    output_pdb: Path | str = "multiresTriag.pdb",
    system_xml: Path | str = "system.xml",
    forcefield_files: Sequence[str] | None = None,
    shell_center: Sequence[float] = (4.0, 4.0, 4.0),
    shell_radius: float = 2.0,
    shell_cutoff: float = 0.4,
    extra_space: float = 0.15,
    force_constant=3000.0 * unit.kilojoules_per_mole / unit.nanometer**2,
    num_subdivisions: int = 3,
    plot_histogram: bool = False,
) -> List[float]:
    """Prepare a PDB/System pair with a coarse-grained triangular boundary."""

    forcefield_files = forcefield_files or ("amber99sb.xml", "spce.xml")
    input_path = Path(input_pdb)
    output_pdb_path = Path(output_pdb)
    system_xml_path = Path(system_xml)

    LOGGER.info("Reading %s", input_path)
    pdb = PDBFile(str(input_path))
    forcefield = ForceField(*forcefield_files)
    modeller = Modeller(pdb.topology, pdb.positions)

    LOGGER.info("Input system contains %d atoms", pdb.topology.getNumAtoms())
    pos_array = clean_openmm_positions(pdb.positions)
    atoms = list(pdb.topology.atoms())
    to_remove = []

    for idx, vec in enumerate(pos_array):
        if distance(vec, shell_center) > shell_radius:
            residue = atoms[idx].residue
            if residue not in to_remove:
                to_remove.append(residue)

    if to_remove:
        LOGGER.info("Removing %d residues outside free-molecule sphere", len(to_remove))
        modeller.delete(to_remove)

    mod_top = modeller.getTopology()
    mod_pos = modeller.getPositions()
    LOGGER.info("Number of atoms after delete: %d", mod_top.getNumAtoms())

    shift = Vec3(*shell_center) * unit.nanometer
    translated_positions = [pos - shift for pos in mod_pos]

    buffer = io.StringIO()
    PDBFile.writeFile(mod_top, translated_positions, buffer, keepIds=True)
    base_lines = buffer.getvalue().splitlines(keepends=True)
    while base_lines and base_lines[-1].startswith(("ENDMDL", "END")):
        base_lines.pop()

    base_atom_count = mod_top.getNumAtoms()
    base_residue_count = mod_top.getNumResidues()

    verts, faces = icosahedron()
    for _ in range(num_subdivisions):
        verts, faces = subdivide(verts, faces)
    verts = deduplicate_vertices(verts)

    xs, ys, zs = [], [], []
    for vert in verts:
        xs.append(vert[0] * (shell_radius + extra_space))
        ys.append(vert[1] * (shell_radius + extra_space))
        zs.append(vert[2] * (shell_radius + extra_space))

    LOGGER.info("Number of molecules in boundary: %d", len(xs))

    hyd1x, hyd1y, hyd1z = [], [], []
    hyd2x, hyd2y, hyd2z = [], [], []

    r = 0.1
    phi = 0.0
    for idx in range(len(xs)):
        theta = 2 * pi if idx % 2 else pi
        x = r * sin(theta) * cos(phi)
        y = r * sin(theta) * sin(phi)
        z = r * cos(theta)

        x2 = r * sin(theta + 1.823) * cos(phi)
        y2 = r * sin(theta + 1.823) * sin(phi)
        z2 = r * cos(theta + 1.823)

        hyd1x.append(xs[idx] + x)
        hyd1y.append(ys[idx] + y)
        hyd1z.append(zs[idx] + z)
        hyd2x.append(xs[idx] + x2)
        hyd2y.append(ys[idx] + y2)
        hyd2z.append(zs[idx] + z2)

    atom_serial = base_atom_count + 1
    residue_serial = base_residue_count + 1
    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb_path.open("w") as handle:
        handle.writelines(base_lines)
        for idx in range(len(xs)):
            handle.write(
                _format_pdb_line(
                    "ATOM",
                    atom_serial,
                    "OW",
                    "SOL",
                    residue_serial + idx,
                    xs[idx] * 10,
                    ys[idx] * 10,
                    zs[idx] * 10,
                )
            )
            handle.write(
                _format_pdb_line(
                    "ATOM",
                    atom_serial + 1,
                    "HW1",
                    "SOL",
                    residue_serial + idx,
                    hyd1x[idx] * 10,
                    hyd1y[idx] * 10,
                    hyd1z[idx] * 10,
                )
            )
            handle.write(
                _format_pdb_line(
                    "ATOM",
                    atom_serial + 2,
                    "HW2",
                    "SOL",
                    residue_serial + idx,
                    hyd2x[idx] * 10,
                    hyd2y[idx] * 10,
                    hyd2z[idx] * 10,
                )
            )
            atom_serial += 3
        handle.write("ENDMDL\nEND\n")

    LOGGER.info("Total number of atoms after boundary: %d", atom_serial - 1)

    final_pdb = PDBFile(str(output_pdb_path))
    final_top = final_pdb.topology

    system = forcefield.createSystem(
        final_top,
        nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=1 * unit.nanometer,
        removeCMMotion=True,
    )

    enm_spring = HarmonicBondForce()
    system.addForce(enm_spring)

    boundary_oxygens = [(xs[i], ys[i], zs[i]) for i in range(len(xs))]
    dists: List[float] = []
    offset = base_atom_count

    for i in range(len(boundary_oxygens)):
        for j in range(i + 1, len(boundary_oxygens)):
            d = distance(boundary_oxygens[i], boundary_oxygens[j])
            if d < shell_cutoff:
                enm_spring.addBond(
                    offset + i * 3,
                    offset + j * 3,
                    d * unit.nanometer,
                    force_constant,
                )
                dists.append(d)

    meanradius = sum(distance((x, y, z), (0.0, 0.0, 0.0)) for x, y, z in boundary_oxygens)
    meanradius /= len(boundary_oxygens)
    LOGGER.info("Boundary mean radius: %.3f nm", meanradius)
    if dists:
        LOGGER.info(
            "Spring distance stats -> mean: %.3f nm, min: %.3f, max: %.3f",
            sum(dists) / len(dists),
            min(dists),
            max(dists),
        )
    else:
        LOGGER.warning("No elastic network springs were added to the boundary shell")

    system_xml_path.parent.mkdir(parents=True, exist_ok=True)
    with system_xml_path.open("w") as handle:
        handle.write(XmlSerializer.serialize(system))

    if plot_histogram:
        try:
            from matplotlib import pyplot as plt
        except ImportError:  # pragma: no cover - optional dependency
            LOGGER.warning("Matplotlib is not installed; cannot display histogram")
        else:
            plt.figure()
            plt.hist(dists, bins=50, range=(0.25, 0.6), facecolor="#708090")
            plt.xlabel(r"$r^0$ [nm]")
            plt.ylabel("Number of Springs")
            plt.show()

    return dists


def _format_pdb_line(
    record: str,
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_id: int,
    x: float,
    y: float,
    z: float,
    occupancy: float = 1.00,
    temp_factor: float = 0.00,
) -> str:
    """Format a simple ATOM/HETATM PDB line."""

    return (
        f"{record:<6}{serial:5d} {atom_name:^4}{residue_name:>4} "
        f"{residue_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{temp_factor:6.2f}\n"
    )
