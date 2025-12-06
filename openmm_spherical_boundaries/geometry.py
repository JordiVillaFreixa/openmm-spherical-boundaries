"""Simple polyhedra helpers used for building spherical boundaries."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .utils import distance

Vector = Tuple[float, float, float]
Face = Tuple[int, int, int]


def icosahedron() -> Tuple[List[Vector], List[Face]]:
    """Return vertices/faces for a unit icosahedron centered at the origin."""

    faces: List[Face] = [
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 5),
        (0, 5, 1),
        (11, 6, 7),
        (11, 7, 8),
        (11, 8, 9),
        (11, 9, 10),
        (11, 10, 6),
        (1, 2, 6),
        (2, 3, 7),
        (3, 4, 8),
        (4, 5, 9),
        (5, 1, 10),
        (6, 7, 2),
        (7, 8, 3),
        (8, 9, 4),
        (9, 10, 5),
        (10, 6, 1),
    ]
    verts: List[Vector] = [
        (0.000, 0.000, 1.000),
        (0.894, 0.000, 0.447),
        (0.276, 0.851, 0.447),
        (-0.724, 0.526, 0.447),
        (-0.724, -0.526, 0.447),
        (0.276, -0.851, 0.447),
        (0.724, 0.526, -0.447),
        (-0.276, 0.851, -0.447),
        (-0.894, 0.000, -0.447),
        (-0.276, -0.851, -0.447),
        (0.724, -0.526, -0.447),
        (0.000, 0.000, -1.000),
    ]
    return verts, faces


def octahedron() -> Tuple[List[Vector], List[Face]]:
    """Return vertices/faces for a unit octahedron centered at the origin."""

    f = 2**0.5 / 2.0
    verts: List[Vector] = [
        (0.0, -1.0, 0.0),
        (-f, 0.0, f),
        (f, 0.0, f),
        (f, 0.0, -f),
        (-f, 0.0, -f),
        (0.0, 1.0, 0.0),
    ]
    faces: List[Face] = [
        (0, 2, 1),
        (0, 3, 2),
        (0, 4, 3),
        (0, 1, 4),
        (5, 1, 2),
        (5, 2, 3),
        (5, 3, 4),
        (5, 4, 1),
    ]
    return verts, faces


def subdivide(verts: List[Vector], faces: List[Face]) -> Tuple[List[Vector], List[Face]]:
    """Subdivide each triangular face into four smaller triangles."""

    result_verts = list(verts)
    result_faces = list(faces)
    triangles = len(faces)

    for face_index in range(triangles):
        face = result_faces[face_index]
        a, b, c = (result_verts[idx] for idx in face)

        i = len(result_verts)
        result_verts.append(_normalize(_midpoint(a, b)))
        result_verts.append(_normalize(_midpoint(b, c)))
        result_verts.append(_normalize(_midpoint(a, c)))

        ab = i
        bc = i + 1
        ac = i + 2

        result_faces.append((ab, bc, ac))
        result_faces.append((face[0], ab, ac))
        result_faces.append((ab, face[1], bc))
        result_faces[face_index] = (ac, bc, face[2])

    return result_verts, result_faces


def deduplicate_vertices(verts: Sequence[Vector], tol: float = 1e-10) -> List[Vector]:
    """Remove duplicate vertices introduced during subdivision."""

    unique: List[Vector] = []
    for candidate in verts:
        if not any(distance(candidate, existing) < tol for existing in unique):
            unique.append(candidate)
    return unique


def _midpoint(a: Vector, b: Vector) -> Vector:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _normalize(vec: Vector) -> Vector:
    length = (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5
    if length == 0:
        return vec
    return (vec[0] / length, vec[1] / length, vec[2] / length)
