"""Utility for building a standalone spherical water droplet."""

from __future__ import annotations

import logging
from math import cos, pi, sin
from pathlib import Path
from typing import Sequence

from openmm import HarmonicBondForce, Vec3, XmlSerializer, unit
from openmm.app import CutoffNonPeriodic, Element, ForceField, Modeller, PDBFile, Topology

from .geometry import deduplicate_vertices, icosahedron, subdivide
from .utils import clean_openmm_positions, distance

LOGGER = logging.getLogger(__name__)

DEFAULT_FORCEFIELD = ("amber99sb.xml", "spce.xml")


def prepare_water_droplet(
    radius: float,
    output_pdb: Path | str = "droplet.pdb",
    system_xml: Path | str = "droplet.xml",
    forcefield_files: Sequence[str] | None = None,
    solvent_model: str = "spce",
    padding: float = 0.3,
    shell_thickness: float = 0.3,
    shell_cutoff: float = 0.6,
    force_constant=None,
    boundary_mode: str = "triangular",
    extra_space: float = 0.15,
    num_subdivisions: int = 3,
) -> int:
    """Build a spherical water droplet and add elastic boundary restraints.

    boundary_mode:
        - ``molten``: current behavior, harmonic springs between waters in the outer shell
        - ``triangular``: single-layer icosahedral net with near-uniform spacing (~0.32 nm)
    """

    if radius <= 0:
        raise ValueError("Radius must be positive.")

    boundary_mode = boundary_mode.lower()
    if boundary_mode not in {"molten", "triangular"}:
        raise ValueError("boundary_mode must be 'molten' or 'triangular'")

    if shell_thickness <= 0 and boundary_mode == "molten":
        raise ValueError("Shell thickness must be positive.")
    if shell_cutoff <= 0:
        raise ValueError("Shell cutoff must be positive.")
    if extra_space < 0:
        raise ValueError("extra_space must be non-negative.")
    if num_subdivisions < 0:
        raise ValueError("num_subdivisions must be non-negative.")

    # Default force constant depends on the boundary type (thesis values)
    if force_constant is None:
        base_force_constant = 3000.0 if boundary_mode == "triangular" else 1000.0
    else:
        base_force_constant = force_constant
    if not isinstance(base_force_constant, unit.Quantity):
        base_force_constant = (
            base_force_constant * unit.kilojoules_per_mole / unit.nanometer**2
        )

    if forcefield_files is None:
        forcefield_files = DEFAULT_FORCEFIELD
    forcefield_tuple = tuple(forcefield_files)
    if set(forcefield_tuple) != set(DEFAULT_FORCEFIELD):
        raise NotImplementedError(
            "Droplet generation currently supports only amber99sb.xml + spce.xml."
        )

    forcefield = ForceField(*forcefield_tuple)
    topology, positions = _single_water()
    modeller = Modeller(topology, positions)

    box_edge = 2.0 * (radius + padding)
    LOGGER.info(
        "Packing water using %s model into %.2f nm cube before trimming", solvent_model, box_edge
    )
    modeller.addSolvent(
        forcefield,
        model=solvent_model,
        boxSize=Vec3(box_edge, box_edge, box_edge),
    )

    trimmed_count = _trim_to_radius(modeller, radius)
    LOGGER.info("Trimmed droplet contains %d water molecules", trimmed_count)

    # Use mode-appropriate cutoff (default of 0.6 is best for molten; 0.4 was tested for triangular)
    effective_cutoff = shell_cutoff
    if boundary_mode == "triangular" and shell_cutoff == 0.6:
        effective_cutoff = 0.4
        LOGGER.info("Triangular boundary detected; using shell_cutoff=0.4 nm to match tested net")

    if boundary_mode == "molten":
        boundary_atoms = _identify_boundary_oxygens(modeller, radius, shell_thickness)
        LOGGER.info("Boundary shell contains %d oxygen atoms", len(boundary_atoms))
        system = forcefield.createSystem(
            modeller.getTopology(),
            nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=1 * unit.nanometer,
            removeCMMotion=True,
        )
        _add_boundary_forces(system, boundary_atoms, effective_cutoff, base_force_constant)
    else:
        offset = modeller.getTopology().getNumAtoms()
        enm_force = _add_triangular_boundary(
            modeller=modeller,
            radius=radius,
            shell_cutoff=effective_cutoff,
            force_constant=base_force_constant,
            extra_space=extra_space,
            num_subdivisions=num_subdivisions,
            starting_index=offset,
        )
        system = forcefield.createSystem(
            modeller.getTopology(),
            nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=1 * unit.nanometer,
            removeCMMotion=True,
        )
        system.addForce(enm_force)

    output_pdb_path = Path(output_pdb)
    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    with output_pdb_path.open("w") as handle:
        PDBFile.writeFile(modeller.getTopology(), modeller.getPositions(), handle, keepIds=True)

    system_xml_path = Path(system_xml)
    system_xml_path.parent.mkdir(parents=True, exist_ok=True)
    with system_xml_path.open("w") as handle:
        handle.write(XmlSerializer.serialize(system))

    return trimmed_count


def _single_water() -> tuple[Topology, list[Vec3]]:
    """Return a topology and coordinates for a single SPC/E-like water molecule."""

    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    element_o = Element.getBySymbol("O")
    element_h = Element.getBySymbol("H")
    atom_o = topology.addAtom("O", element_o, residue)
    atom_h1 = topology.addAtom("H1", element_h, residue)
    atom_h2 = topology.addAtom("H2", element_h, residue)
    topology.addBond(atom_o, atom_h1)
    topology.addBond(atom_o, atom_h2)

    positions = [
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.09572, 0.0, 0.0),
        Vec3(-0.031906, 0.092386, 0.0),
    ]
    return topology, unit.Quantity(positions, unit.nanometer)


def _trim_to_radius(modeller: Modeller, radius: float) -> int:
    """Remove residues whose oxygen atoms lie outside the requested radius."""

    pos_array = clean_openmm_positions(modeller.getPositions())
    residues_to_remove = []
    for residue in modeller.getTopology().residues():
        oxygen_index = None
        for atom in residue.atoms():
            if getattr(atom.element, "symbol", None) == "O":
                oxygen_index = atom.index
                break
        if oxygen_index is None:
            continue
        coord = pos_array[oxygen_index]
        if distance(coord, (0.0, 0.0, 0.0)) > radius:
            residues_to_remove.append(residue)

    if residues_to_remove:
        LOGGER.info("Removing %d residues outside %.2f nm sphere", len(residues_to_remove), radius)
        modeller.delete(residues_to_remove)

    return modeller.getTopology().getNumResidues()


def _identify_boundary_oxygens(
    modeller: Modeller,
    radius: float,
    shell_thickness: float,
) -> list[tuple[int, list[float]]]:
    """Return (index, coords) for oxygen atoms within the boundary shell."""

    inner_radius = max(0.0, radius - shell_thickness)
    pos_array = clean_openmm_positions(modeller.getPositions())
    boundary: list[tuple[int, list[float]]] = []
    for atom, coord in zip(modeller.getTopology().atoms(), pos_array):
        if getattr(atom.element, "symbol", None) != "O":
            continue
        dist_to_center = distance(coord, (0.0, 0.0, 0.0))
        if inner_radius <= dist_to_center <= radius:
            boundary.append((atom.index, coord))
    return boundary


def _add_boundary_forces(
    system,
    boundary_atoms: list[tuple[int, list[float]]],
    shell_cutoff: float,
    force_constant,
) -> None:
    """Add harmonic bonds between nearby boundary oxygens."""

    if not boundary_atoms:
        LOGGER.warning("No boundary oxygen atoms were detected; skipping boundary springs")
        return

    enm_force = HarmonicBondForce()
    distances = []
    for i, (idx_i, coord_i) in enumerate(boundary_atoms):
        for idx_j, coord_j in boundary_atoms[i + 1 :]:
            d = distance(coord_i, coord_j)
            if d < shell_cutoff:
                enm_force.addBond(
                    idx_i,
                    idx_j,
                    d * unit.nanometer,
                    force_constant,
                )
                distances.append(d)

    if not distances:
        LOGGER.warning(
            "No elastic boundary bonds met the cutoff of %.2f nm; increase shell thickness or cutoff",
            shell_cutoff,
        )
        return

    mean_d = sum(distances) / len(distances)
    LOGGER.info(
        "Added %d boundary springs (mean %.3f nm, min %.3f, max %.3f)",
        len(distances),
        mean_d,
        min(distances),
        max(distances),
    )
    system.addForce(enm_force)


def _add_triangular_boundary(
    modeller: Modeller,
    radius: float,
    shell_cutoff: float,
    force_constant,
    extra_space: float,
    num_subdivisions: int,
    starting_index: int,
) -> HarmonicBondForce:
    """Add a single-layer icosahedral net of waters and return the boundary force."""

    verts, faces = icosahedron()
    for _ in range(num_subdivisions):
        verts, faces = subdivide(verts, faces)
    verts = deduplicate_vertices(verts)

    shell_radius = radius + extra_space
    oxygens = [
        (vert[0] * shell_radius, vert[1] * shell_radius, vert[2] * shell_radius) for vert in verts
    ]
    LOGGER.info("Triangular boundary molecules: %d", len(oxygens))

    # Simple orientation of hydrogens on the sphere (as in triangular_model)
    r = 0.1
    phi = 0.0
    hydrogens = []
    for idx, (x, y, z) in enumerate(oxygens):
        theta = 2 * pi if idx % 2 else pi
        x1 = r * sin(theta) * cos(phi)
        y1 = r * sin(theta) * sin(phi)
        z1 = r * cos(theta)
        x2 = r * sin(theta + 1.823) * cos(phi)
        y2 = r * sin(theta + 1.823) * sin(phi)
        z2 = r * cos(theta + 1.823)
        hydrogens.append(((x + x1, y + y1, z + z1), (x + x2, y + y2, z + z2)))

    element_o = Element.getBySymbol("O")
    element_h = Element.getBySymbol("H")

    topo = Topology()
    chain = topo.addChain()
    boundary_positions: list[Vec3] = []

    for idx, (ox, oy, oz) in enumerate(oxygens):
        residue = topo.addResidue("SOL", chain)
        atom_o = topo.addAtom("O", element_o, residue)
        atom_h1 = topo.addAtom("H1", element_h, residue)
        atom_h2 = topo.addAtom("H2", element_h, residue)
        topo.addBond(atom_o, atom_h1)
        topo.addBond(atom_o, atom_h2)

        (h1x, h1y, h1z), (h2x, h2y, h2z) = hydrogens[idx]
        boundary_positions.extend(
            [
                Vec3(ox, oy, oz),
                Vec3(h1x, h1y, h1z),
                Vec3(h2x, h2y, h2z),
            ]
        )

    modeller.add(topo, unit.Quantity(boundary_positions, unit.nanometer))

    enm_force = HarmonicBondForce()
    distances: list[float] = []
    for i in range(len(oxygens)):
        for j in range(i + 1, len(oxygens)):
            d = distance(oxygens[i], oxygens[j])
            if d < shell_cutoff:
                enm_force.addBond(
                    starting_index + i * 3,
                    starting_index + j * 3,
                    d * unit.nanometer,
                    force_constant,
                )
                distances.append(d)

    if distances:
        mean_d = sum(distances) / len(distances)
        LOGGER.info(
            "Added %d triangular boundary springs (mean %.3f nm, min %.3f, max %.3f)",
            len(distances),
            mean_d,
            min(distances),
            max(distances),
        )
    else:
        LOGGER.warning("No triangular boundary springs added; check cutoff or subdivisions")

    return enm_force
