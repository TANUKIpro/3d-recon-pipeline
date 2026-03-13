"""2D polygon triangulation and UV projection helpers."""

from __future__ import annotations

import numpy as np


def _polygon_area_2d(points_2d: np.ndarray) -> float:
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    bc = c - b
    return float(ab[0] * bc[1] - ab[1] * bc[0])


def _point_in_tri_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-10) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) <= eps:
        return False

    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


def _triangulate_polygon_ear_clip(loop_uv: np.ndarray) -> list[tuple[int, int, int]]:
    n = int(loop_uv.shape[0])
    if n < 3:
        return []

    area = _polygon_area_2d(loop_uv)
    if abs(area) < 1e-12:
        return []
    orientation = 1.0 if area > 0.0 else -1.0

    idx = list(range(n))
    triangles: list[tuple[int, int, int]] = []
    max_iter = n * n
    steps = 0

    while len(idx) > 3 and steps < max_iter:
        ear_found = False
        m = len(idx)
        for i in range(m):
            i_prev = idx[(i - 1) % m]
            i_curr = idx[i]
            i_next = idx[(i + 1) % m]

            a = loop_uv[i_prev]
            b = loop_uv[i_curr]
            c = loop_uv[i_next]

            if _cross2(a, b, c) * orientation <= 1e-12:
                continue

            any_inside = False
            for j in idx:
                if j in {i_prev, i_curr, i_next}:
                    continue
                if _point_in_tri_2d(loop_uv[j], a, b, c):
                    any_inside = True
                    break
            if any_inside:
                continue

            if orientation > 0:
                triangles.append((i_prev, i_curr, i_next))
            else:
                triangles.append((i_prev, i_next, i_curr))
            del idx[i]
            ear_found = True
            break

        if not ear_found:
            return []
        steps += 1

    if len(idx) == 3:
        if orientation > 0:
            triangles.append((idx[0], idx[1], idx[2]))
        else:
            triangles.append((idx[0], idx[2], idx[1]))

    return triangles


def _loop_projection_uv(loop_points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(normal, axis))) > 0.92:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    tangent_u = np.cross(normal, axis)
    tangent_u = tangent_u / max(float(np.linalg.norm(tangent_u)), 1e-12)
    tangent_v = np.cross(normal, tangent_u)
    tangent_v = tangent_v / max(float(np.linalg.norm(tangent_v)), 1e-12)

    centered = loop_points - loop_points.mean(axis=0)
    u = centered @ tangent_u
    v = centered @ tangent_v
    return np.column_stack((u, v))
