"""Shared helper utilities."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from openmm import Vec3, unit

VectorLike = Sequence[float]


def clean_openmm_positions(positions: Iterable[Vec3]) -> List[List[float]]:
    """Convert OpenMM position objects to bare floats measured in nanometers."""

    cleaned: List[List[float]] = []
    for vec in positions:
        try:
            value = vec.value_in_unit(unit.nanometer)
        except AttributeError:
            value = vec
        cleaned.append([float(value[0]), float(value[1]), float(value[2])])
    return cleaned


def as_quantity(coords: VectorLike) -> Vec3:
    """Return a Vec3 quantity expressed in nanometers."""

    return Vec3(*coords) * unit.nanometer


def distance(a: VectorLike, b: VectorLike) -> float:
    """Euclidean distance between Cartesian coordinates (in nanometers)."""

    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def translate(coords: VectorLike, shift: VectorLike) -> Tuple[float, float, float]:
    """Translate coords by subtracting the supplied shift vector."""

    return (coords[0] - shift[0], coords[1] - shift[1], coords[2] - shift[2])
