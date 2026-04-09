"""Custom collagen metrics computed directly in Python."""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from typing import Iterable

import numpy as np
from scipy import ndimage


NEIGHBOR_STEPS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def zhang_suen_skeletonize(mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask to a 1-pixel skeleton using Zhang-Suen."""
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    changed = True

    while changed:
        changed = False
        for step in (0, 1):
            markers = []
            rows, cols = np.where(padded == 1)
            for r, c in zip(rows, cols):
                if r == 0 or c == 0 or r == padded.shape[0] - 1 or c == padded.shape[1] - 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors(padded, r, c)
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                transitions = _transition_count(neighbors)
                count = sum(neighbors)
                if not (2 <= count <= 6 and transitions == 1):
                    continue
                if step == 0:
                    if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
                        continue
                else:
                    if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
                        continue
                markers.append((r, c))

            if markers:
                changed = True
                for r, c in markers:
                    padded[r, c] = 0

    return padded[1:-1, 1:-1].astype(bool)


def compute_python_metrics(mask: np.ndarray, pixel_size_um: float = 1.0) -> dict[str, float]:
    """Compute custom and TWOMBLI-like metrics from a binary mask."""
    skeleton = zhang_suen_skeletonize(mask)
    distance_map = ndimage.distance_transform_edt(mask) * float(pixel_size_um)

    component_paths = _extract_component_paths(skeleton)
    tortuosities = []
    curvatures_30 = []
    curvatures_40 = []
    path_lengths = []

    for path in component_paths:
        path_length = _path_length(path) * float(pixel_size_um)
        if path_length <= 0:
            continue
        path_lengths.append(path_length)
        chord = _endpoint_distance(path) * float(pixel_size_um)
        tortuosities.append(chord / path_length if path_length > 0 else np.nan)
        curvatures_30.append(_path_curvature(path, scale_um=30.0, pixel_size_um=float(pixel_size_um)))
        curvatures_40.append(_path_curvature(path, scale_um=40.0, pixel_size_um=float(pixel_size_um)))

    width_values = distance_map[skeleton]
    endpoints, branch_points = _count_special_points(skeleton)
    total_length = float(sum(path_lengths))

    return {
        "mask_area_fraction": float(mask.mean()),
        "fibre_length_um": total_length,
        "lacunarity": float(compute_lacunarity(mask)),
        "endpoints": float(endpoints),
        "branch_points": float(branch_points),
        "hyphal_growth_unit_um": float(total_length / endpoints) if endpoints else np.nan,
        "curvature_30um_deg": _safe_nanmean(curvatures_30),
        "curvature_40um_deg": _safe_nanmean(curvatures_40),
        "fractal_dimension_boxcount": float(box_counting_dimension(mask)),
        "fibre_width_um": float(np.nanmean(width_values)) if width_values.size else np.nan,
        "length_tortuosity": _safe_nanmean(tortuosities),
        "skeleton_pixels": float(np.count_nonzero(skeleton)),
        "connected_components": float(len(component_paths)),
    }


def compute_lacunarity(mask: np.ndarray, box_sizes: Iterable[int] = (2, 4, 8, 16, 32)) -> float:
    """Estimate lacunarity from occupancy variance across box sizes."""
    mask_float = mask.astype(np.float32)
    values = []
    for size in box_sizes:
        if size > min(mask.shape):
            continue
        summed = ndimage.uniform_filter(mask_float, size=size, mode="constant") * (size * size)
        flattened = summed.ravel()
        mean = flattened.mean()
        if mean <= 0:
            continue
        values.append(flattened.var() / (mean * mean) + 1.0)
    return float(np.nanmean(values)) if values else np.nan


def box_counting_dimension(mask: np.ndarray, box_sizes: Iterable[int] = (2, 4, 8, 16, 32, 64)) -> float:
    """Estimate fractal dimension with a simple box-counting method."""
    counts = []
    scales = []
    for size in box_sizes:
        if size > min(mask.shape):
            continue
        count = _count_nonempty_boxes(mask, size)
        if count <= 0:
            continue
        counts.append(np.log(count))
        scales.append(np.log(1.0 / size))

    if len(counts) < 2:
        return np.nan
    slope, _ = np.polyfit(scales, counts, 1)
    return float(slope)


def _count_nonempty_boxes(mask: np.ndarray, size: int) -> int:
    rows = int(np.ceil(mask.shape[0] / size))
    cols = int(np.ceil(mask.shape[1] / size))
    padded = np.zeros((rows * size, cols * size), dtype=bool)
    padded[: mask.shape[0], : mask.shape[1]] = mask
    reshaped = padded.reshape(rows, size, cols, size)
    return int(np.count_nonzero(reshaped.any(axis=(1, 3))))


def _count_special_points(skeleton: np.ndarray) -> tuple[int, int]:
    endpoints = 0
    branch_points = 0
    rows, cols = np.where(skeleton)
    for r, c in zip(rows, cols):
        degree = sum(
            1
            for dr, dc in NEIGHBOR_STEPS
            if 0 <= r + dr < skeleton.shape[0]
            and 0 <= c + dc < skeleton.shape[1]
            and skeleton[r + dr, c + dc]
        )
        if degree == 1:
            endpoints += 1
        elif degree >= 3:
            branch_points += 1
    return endpoints, branch_points


def _extract_component_paths(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    labels, count = ndimage.label(skeleton)
    paths = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        coords = list(map(tuple, np.argwhere(component)))
        if len(coords) < 2:
            continue
        endpoints = [coord for coord in coords if _degree(component, coord) == 1]
        if len(endpoints) >= 2:
            best_path = _longest_shortest_path(component, endpoints)
            if best_path:
                paths.append(best_path)
        else:
            paths.append(_ordered_component_walk(component, coords[0]))
    return paths


def _longest_shortest_path(component: np.ndarray, endpoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    best_path = []
    best_length = -1.0
    for start in endpoints:
        parents, distances = _dijkstra_paths(component, start)
        for end in endpoints:
            if end == start or end not in distances:
                continue
            length = distances[end]
            if length > best_length:
                best_length = length
                best_path = _reconstruct_path(parents, end)
    return best_path


def _ordered_component_walk(component: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    visited = set()
    order = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in _neighbors_coords(component, node):
            if neighbor not in visited:
                stack.append(neighbor)
    return order


def _dijkstra_paths(component: np.ndarray, start: tuple[int, int]):
    heap = [(0.0, start)]
    parents = {start: None}
    distances = {start: 0.0}

    while heap:
        distance, node = heappop(heap)
        if distance > distances[node]:
            continue
        for neighbor in _neighbors_coords(component, node):
            new_distance = distance + _step_length(node, neighbor)
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                parents[neighbor] = node
                heappush(heap, (new_distance, neighbor))
    return parents, distances


def _reconstruct_path(parents: dict[tuple[int, int], tuple[int, int] | None], end: tuple[int, int]):
    path = deque()
    node = end
    while node is not None:
        path.appendleft(node)
        node = parents[node]
    return list(path)


def _degree(component: np.ndarray, coord: tuple[int, int]) -> int:
    return sum(1 for _ in _neighbors_coords(component, coord))


def _neighbors_coords(component: np.ndarray, coord: tuple[int, int]):
    r, c = coord
    for dr, dc in NEIGHBOR_STEPS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < component.shape[0] and 0 <= nc < component.shape[1] and component[nr, nc]:
            yield (nr, nc)


def _path_length(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    return float(sum(_step_length(a, b) for a, b in zip(path[:-1], path[1:])))


def _endpoint_distance(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    start = np.asarray(path[0], dtype=np.float64)
    end = np.asarray(path[-1], dtype=np.float64)
    return float(np.linalg.norm(end - start))


def _path_curvature(path: list[tuple[int, int]], scale_um: float, pixel_size_um: float) -> float:
    if len(path) < 3:
        return np.nan

    window = max(1, int(round(scale_um / max(pixel_size_um, 1e-6))))
    if len(path) < (2 * window + 1):
        return np.nan

    coords = np.asarray(path, dtype=np.float64)
    angles = []
    for idx in range(window, len(coords) - window):
        v1 = coords[idx] - coords[idx - window]
        v2 = coords[idx + window] - coords[idx]
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm <= 0:
            continue
        cosine = np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosine)))
    return float(np.nanmean(angles)) if angles else np.nan


def _step_length(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _neighbors(image: np.ndarray, r: int, c: int) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        image[r - 1, c],
        image[r - 1, c + 1],
        image[r, c + 1],
        image[r + 1, c + 1],
        image[r + 1, c],
        image[r + 1, c - 1],
        image[r, c - 1],
        image[r - 1, c - 1],
    )


def _transition_count(neighbors: list[int]) -> int:
    wrapped = neighbors + [neighbors[0]]
    return sum((a == 0 and b == 1) for a, b in zip(wrapped[:-1], wrapped[1:]))


def _safe_nanmean(values: list[float]) -> float:
    if not values:
        return np.nan
    array = np.asarray(values, dtype=np.float64)
    if np.isnan(array).all():
        return np.nan
    return float(np.nanmean(array))
