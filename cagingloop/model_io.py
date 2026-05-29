from __future__ import annotations

from pathlib import Path

import numpy as np


def _parse_face_vertex(token: str) -> int:
    return int(token.split("/")[0]) - 1


def load_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                ids = [_parse_face_vertex(token) for token in line.split()[1:]]
                if len(ids) < 3:
                    continue
                for i in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[i], ids[i + 1]])

    if not vertices:
        raise ValueError(f"{path} contains no OBJ vertices")
    if not faces:
        raise ValueError(f"{path} contains no OBJ faces")
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must be an N x 3 array")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be an M x 3 array")

    normals = np.zeros_like(vertices, dtype=float)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for face, normal in zip(faces, face_normals, strict=True):
        normals[face] += normal

    lengths = np.linalg.norm(normals, axis=1)
    center = vertices.mean(axis=0)
    fallback = vertices - center
    fallback_lengths = np.linalg.norm(fallback, axis=1)
    fallback_lengths[fallback_lengths == 0.0] = 1.0
    fallback = fallback / fallback_lengths[:, None]

    zero = lengths == 0.0
    lengths[zero] = 1.0
    normals = normals / lengths[:, None]
    normals[zero] = fallback[zero]
    return normals


def load_obj_point_cloud(path: str | Path, *, max_points: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, faces = load_obj_mesh(path)
    normals = compute_vertex_normals(vertices, faces)
    if max_points is not None and max_points > 0 and len(vertices) > max_points:
        ids = np.linspace(0, len(vertices) - 1, max_points, dtype=int)
        vertices = vertices[ids]
        normals = normals[ids]
    return vertices, normals, faces
