"""Create a standard periodic reference simulation system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from openmm import XmlSerializer, unit
from openmm.app import ForceField, PDBFile, PME

LOGGER = logging.getLogger(__name__)


def prepare_reference_system(
    input_pdb: Path | str,
    output_pdb: Path | str = "multiresRef.pdb",
    system_xml: Path | str = "system_ref.xml",
    forcefield_files: Sequence[str] | None = None,
) -> None:
    """Write a reference PDB/XML pair using periodic boundary conditions."""

    forcefield_files = forcefield_files or ("amber99sb.xml", "spce.xml")
    input_path = Path(input_pdb)
    output_pdb_path = Path(output_pdb)
    system_xml_path = Path(system_xml)

    LOGGER.info("Reading %s", input_path)
    pdb = PDBFile(str(input_path))
    forcefield = ForceField(*forcefield_files)

    LOGGER.info("Input system contains %d atoms", pdb.topology.getNumAtoms())

    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb_path.open("w") as handle:
        PDBFile.writeFile(pdb.topology, pdb.positions, handle, keepIds=True)

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1 * unit.nanometer,
        removeCMMotion=True,
    )

    system_xml_path.parent.mkdir(parents=True, exist_ok=True)
    with system_xml_path.open("w") as handle:
        handle.write(XmlSerializer.serialize(system))

    LOGGER.info("Reference system written to %s and %s", output_pdb_path, system_xml_path)
