"""Preparation routine for multi-resolution spherical boundary systems."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from openmm import HarmonicBondForce, XmlSerializer, unit
from openmm.app import CutoffNonPeriodic, ForceField, Modeller, PDBFile

from .utils import clean_openmm_positions, distance

LOGGER = logging.getLogger(__name__)


def prepare_md_system(
    input_pdb: Path | str,
    output_pdb: Path | str = "multiresPrep.pdb",
    system_xml: Path | str = "system.xml",
    forcefield_files: Sequence[str] | None = None,
    shell_center: Sequence[float] = (4.0, 4.0, 4.0),
    shell_radius: float = 2.0,
    shell_thickness: float = 1.0,
    shell_cutoff: float = 0.6,
    force_constant=1000.0 * unit.kilojoules_per_mole / unit.nanometer**2,
    plot_histogram: bool = False,
) -> Tuple[List[float], PDBFile]:
    """Generate a trimmed PDB and OpenMM system with a spherical elastic network."""

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
    to_remove: List = []

    for idx, vec in enumerate(pos_array):
        d = distance(vec, shell_center)
        if d > shell_radius + shell_thickness:
            residue = atoms[idx].residue
            if residue not in to_remove:
                to_remove.append(residue)

    if to_remove:
        LOGGER.info("Removing %d residues outside spherical region", len(to_remove))
        modeller.delete(to_remove)

    mod_top = modeller.getTopology()
    mod_pos = modeller.getPositions()
    LOGGER.info("Trimmed system contains %d atoms", mod_top.getNumAtoms())

    new_atom_list = list(mod_top.atoms())
    new_pos_array = clean_openmm_positions(mod_pos)

    enm_coords: List[Tuple[int, List[float]]] = []
    for idx, (atom, vec) in enumerate(zip(new_atom_list, new_pos_array)):
        d = distance(vec, shell_center)
        if shell_radius < d < shell_radius + shell_thickness:
            element = getattr(atom.element, "symbol", None)
            if element == "O":
                enm_coords.append((idx, vec))

    LOGGER.info("Elastic network will include %d oxygen atoms", len(enm_coords))

    system = forcefield.createSystem(
        mod_top,
        nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=1 * unit.nanometer,
        removeCMMotion=True,
    )

    enm_spring = HarmonicBondForce()
    system.addForce(enm_spring)
    elastic_distances: List[float] = []

    for i in range(len(enm_coords)):
        idx_i, vec_i = enm_coords[i]
        for j in range(i + 1, len(enm_coords)):
            idx_j, vec_j = enm_coords[j]
            d = distance(vec_i, vec_j)
            if d < shell_cutoff:
                enm_spring.addBond(
                    idx_i,
                    idx_j,
                    d * unit.nanometer,
                    force_constant,
                )
                elastic_distances.append(d)

    if elastic_distances:
        mean_d = sum(elastic_distances) / len(elastic_distances)
        LOGGER.info(
            "Elastic network bonds: %d (mean %.3f nm, min %.3f, max %.3f)",
            len(elastic_distances),
            mean_d,
            min(elastic_distances),
            max(elastic_distances),
        )
    else:
        LOGGER.warning("No elastic network bonds were created")

    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb_path.open("w") as handle:
        PDBFile.writeFile(mod_top, mod_pos, handle, keepIds=True)
    LOGGER.info("Wrote trimmed PDB to %s", output_pdb_path)

    system_xml_path.parent.mkdir(parents=True, exist_ok=True)
    with system_xml_path.open("w") as handle:
        handle.write(XmlSerializer.serialize(system))
    LOGGER.info("Wrote serialized system to %s", system_xml_path)

    if plot_histogram and elastic_distances:
        _plot_histogram(elastic_distances, "Distribution of Spring Equilibrium Distances")

    return elastic_distances


def _plot_histogram(values: List[float], title: str) -> None:
    try:
        from matplotlib import pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("Matplotlib is not installed; cannot display histogram")
        return

    plt.figure()
    plt.hist(values, bins=50, facecolor="#708090")
    plt.xlabel(r"$r^0$ [nm]")
    plt.ylabel("Number of Springs")
    plt.title(title)
    plt.show()
