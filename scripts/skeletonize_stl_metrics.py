#!/usr/bin/env python3
"""Skeletonize a vessel STL and quantify metrics at YZ planes along x."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
CACHE_DIR = WORKSPACE / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

LOCAL_SKELETOR = WORKSPACE / "skeletor"
if (LOCAL_SKELETOR / "skeletor" / "__init__.py").exists():
    sys.path.insert(0, str(LOCAL_SKELETOR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import pandas as pd
import pyvista as pv
import skeletor as sk
import trimesh

_PYVISTA_SCREENSHOT_AVAILABLE: bool | None = None
DISTANCE_UNIT = "um"
AREA_UNIT = "um^2"
BoxBounds = tuple[float, float, float, float, float, float]
HighlightBox = tuple[str, BoxBounds]


def default_output_path(input_path: Path, suffix: str) -> Path:
    if input_path.name.endswith(".stl"):
        return input_path.with_suffix(suffix)
    return input_path.with_name(f"{input_path.name}{suffix}")


def load_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle meshes found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh from {path}, got {type(loaded).__name__}")
    return loaded


def pyvista_mesh(mesh: trimesh.Trimesh) -> pv.PolyData:
    faces = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).ravel()
    return pv.PolyData(np.asarray(mesh.vertices), faces)


def edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def skeletonize_mesh(
    mesh: trimesh.Trimesh,
    waves: int,
    step_size: int,
    radius_method: str,
    n_rays: int,
    cleanup: bool,
) -> sk.skeletonize.base.Skeleton:
    fixed = sk.pre.fix_mesh(mesh, remove_disconnected=False, inplace=False)
    skel = sk.skeletonize.by_wavefront(fixed, waves=waves, step_size=step_size)
    if cleanup:
        skel = sk.post.clean_up(skel, mesh=fixed, inplace=False)

    if radius_method == "ray":
        sk.post.radii(skel, mesh=mesh, method="ray", n_rays=n_rays, fallback="knn")
    else:
        sk.post.radii(skel, mesh=mesh, method="knn")

    return skel


def branch_paths(skel: sk.skeletonize.base.Skeleton) -> list[list[int]]:
    graph = skel.get_graph().to_undirected()
    degree = dict(graph.degree())
    anchors = sorted(node for node, node_degree in degree.items() if node_degree != 2)
    visited_edges: set[tuple[int, int]] = set()
    paths: list[list[int]] = []

    for anchor in anchors:
        for neighbor in sorted(graph.neighbors(anchor)):
            key = edge_key(int(anchor), int(neighbor))
            if key in visited_edges:
                continue

            path = [int(anchor), int(neighbor)]
            visited_edges.add(key)
            previous = int(anchor)
            current = int(neighbor)

            while degree.get(current, 0) == 2:
                next_nodes = [int(node) for node in graph.neighbors(current) if int(node) != previous]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                key = edge_key(current, next_node)
                if key in visited_edges:
                    break
                path.append(next_node)
                visited_edges.add(key)
                previous, current = current, next_node

            paths.append(path)

    for a, b in graph.edges():
        key = edge_key(int(a), int(b))
        if key not in visited_edges:
            paths.append([int(a), int(b)])
            visited_edges.add(key)

    return paths


def local_tortuosity(
    nodes: pd.DataFrame,
    path: list[int],
    edge_index: int,
    degree: dict[int, int],
    window_nodes: int,
) -> float:
    start = edge_index
    end = edge_index + 1

    for _ in range(window_nodes):
        candidate = start - 1
        if candidate < 0 or degree.get(path[candidate], 0) != 2:
            break
        start = candidate

    for _ in range(window_nodes):
        candidate = end + 1
        if candidate >= len(path) or degree.get(path[candidate], 0) != 2:
            break
        end = candidate

    coords = nodes.loc[path[start : end + 1], ["x", "y", "z"]].to_numpy(dtype=float)
    if len(coords) < 2:
        return np.nan

    segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    path_length = float(segment_lengths.sum())
    chord = float(np.linalg.norm(coords[0] - coords[-1]))
    return path_length / chord if chord > np.finfo(float).eps else np.nan


def branch_samples(
    skel: sk.skeletonize.base.Skeleton,
    tortuosity_window_nodes: int,
) -> tuple[pd.DataFrame, dict[tuple[int, int], int]]:
    nodes = skel.swc.set_index("node_id")
    graph = skel.get_graph().to_undirected()
    degree = {int(node): int(node_degree) for node, node_degree in graph.degree()}
    paths = branch_paths(skel)
    edge_to_branch: dict[tuple[int, int], int] = {}
    rows: list[dict[str, float | int]] = []

    for branch_id, path in enumerate(paths):
        if len(path) < 2:
            continue

        for edge_index, (child, parent) in enumerate(zip(path[:-1], path[1:])):
            child = int(child)
            parent = int(parent)
            if not graph.has_edge(child, parent):
                continue
            c0 = nodes.loc[child, ["x", "y", "z"]].to_numpy(dtype=float)
            c1 = nodes.loc[parent, ["x", "y", "z"]].to_numpy(dtype=float)
            r0 = float(nodes.at[child, "radius"])
            r1 = float(nodes.at[parent, "radius"])
            radius = float(np.nanmean([r0, r1]))
            length = float(np.linalg.norm(c0 - c1))
            edge_to_branch[edge_key(child, parent)] = branch_id
            tortuosity = local_tortuosity(
                nodes=nodes,
                path=path,
                edge_index=edge_index,
                degree=degree,
                window_nodes=tortuosity_window_nodes,
            )
            rows.append(
                {
                    "branch_id": branch_id,
                    "x0": float(c0[0]),
                    "x1": float(c1[0]),
                    "y0": float(c0[1]),
                    "y1": float(c1[1]),
                    "z0": float(c0[2]),
                    "z1": float(c1[2]),
                    "r0": r0,
                    "r1": r1,
                    "length": length,
                    "radius": radius,
                    "area": float(np.pi * radius**2),
                    "local_tortuosity": tortuosity,
                }
            )

    samples = pd.DataFrame(rows)
    if samples.empty:
        raise ValueError("Skeleton produced no branch edge samples.")
    return samples, edge_to_branch


def plane_metrics(
    samples: pd.DataFrame,
    mesh: trimesh.Trimesh,
    planes: int,
    axis: str = "x",
) -> pd.DataFrame:
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis)
    if axis_index is None:
        raise ValueError(f"Unsupported plane-normal axis: {axis}")

    positions = np.linspace(float(mesh.bounds[0, axis_index]), float(mesh.bounds[1, axis_index]), planes)
    coordinate0 = samples[f"{axis}0"].to_numpy(dtype=float)
    coordinate1 = samples[f"{axis}1"].to_numpy(dtype=float)
    r0 = samples["r0"].to_numpy(dtype=float)
    r1 = samples["r1"].to_numpy(dtype=float)
    length = samples["length"].to_numpy(dtype=float)
    tort = samples["local_tortuosity"].to_numpy(dtype=float)
    coordinate_min = np.minimum(coordinate0, coordinate1)
    coordinate_max = np.maximum(coordinate0, coordinate1)
    coordinate_delta = coordinate1 - coordinate0
    position_column = f"{axis}_position_um"
    rows = []
    for position in positions:
        crossing = (
            (coordinate_min <= position)
            & (position <= coordinate_max)
            & (np.abs(coordinate_delta) > np.finfo(float).eps)
        )
        if not crossing.any():
            rows.append(
                {
                    position_column: float(position),
                    "branch_crossings": 0,
                    "mean_radius_um": np.nan,
                    "summed_lumen_area_um2": np.nan,
                    "median_local_tortuosity": np.nan,
                    "tortuosity_q25": np.nan,
                    "tortuosity_q75": np.nan,
                }
            )
            continue

        t = (position - coordinate0[crossing]) / coordinate_delta[crossing]
        radii = r0[crossing] + t * (r1[crossing] - r0[crossing])
        areas = np.pi * radii**2
        crossing_lengths = length[crossing]
        crossing_tortuosity = tort[crossing]

        valid_radius = np.isfinite(radii) & np.isfinite(crossing_lengths) & (crossing_lengths > 0)
        mean_radius = (
            float(np.average(radii[valid_radius], weights=crossing_lengths[valid_radius]))
            if valid_radius.any()
            else np.nan
        )
        valid_tortuosity = np.isfinite(crossing_tortuosity)
        if valid_tortuosity.any():
            valid_crossing_tortuosity = crossing_tortuosity[valid_tortuosity]
            median_tortuosity = float(np.median(valid_crossing_tortuosity))
            tortuosity_q25, tortuosity_q75 = np.percentile(valid_crossing_tortuosity, [25, 75])
        else:
            median_tortuosity = np.nan
            tortuosity_q25 = np.nan
            tortuosity_q75 = np.nan
        rows.append(
            {
                position_column: float(position),
                "branch_crossings": int(crossing.sum()),
                "mean_radius_um": mean_radius,
                "summed_lumen_area_um2": float(np.nansum(areas)),
                "median_local_tortuosity": median_tortuosity,
                "tortuosity_q25": float(tortuosity_q25),
                "tortuosity_q75": float(tortuosity_q75),
            }
        )

    return pd.DataFrame(rows)


def plane_tortuosity_points(
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    axis: str,
) -> pd.DataFrame:
    position_column = f"{axis}_position_um"
    if position_column not in metrics:
        raise ValueError(f"Metrics do not contain positions for axis: {axis}")

    coordinate0 = samples[f"{axis}0"].to_numpy(dtype=float)
    coordinate1 = samples[f"{axis}1"].to_numpy(dtype=float)
    coordinate_min = np.minimum(coordinate0, coordinate1)
    coordinate_max = np.maximum(coordinate0, coordinate1)
    coordinate_delta = coordinate1 - coordinate0
    tortuosity = samples["local_tortuosity"].to_numpy(dtype=float)
    valid_tortuosity = np.isfinite(tortuosity)
    point_positions: list[np.ndarray] = []
    point_values: list[np.ndarray] = []

    for position in metrics[position_column].to_numpy(dtype=float):
        crossing = (
            (coordinate_min <= position)
            & (position <= coordinate_max)
            & (np.abs(coordinate_delta) > np.finfo(float).eps)
            & valid_tortuosity
        )
        if crossing.any():
            point_positions.append(np.full(int(crossing.sum()), position))
            point_values.append(tortuosity[crossing])

    if not point_positions:
        return pd.DataFrame(columns=[position_column, "local_tortuosity"])
    return pd.DataFrame(
        {
            position_column: np.concatenate(point_positions),
            "local_tortuosity": np.concatenate(point_values),
        }
    )


def axis_median_tortuosity_thresholds(
    metrics_by_axis: dict[str, pd.DataFrame],
    multiplier: float = 1.2,
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        values = metrics_by_axis[axis]["median_local_tortuosity"].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        mean = float(np.mean(finite_values)) if len(finite_values) else np.nan
        means[axis] = mean
        thresholds[axis] = float(multiplier * mean)
    return means, thresholds


def contiguous_high_tortuosity_intervals(
    metrics: pd.DataFrame,
    axis: str,
    threshold: float,
    axis_bounds: tuple[float, float],
) -> list[tuple[float, float]]:
    position_column = f"{axis}_position_um"
    positions = metrics[position_column].to_numpy(dtype=float)
    tortuosity = metrics["median_local_tortuosity"].to_numpy(dtype=float)
    above_threshold = np.isfinite(tortuosity) & (tortuosity > threshold)
    indices = np.flatnonzero(above_threshold)
    if not len(indices):
        return []

    if len(positions) == 1:
        edges = np.asarray(axis_bounds, dtype=float)
    else:
        midpoints = (positions[:-1] + positions[1:]) / 2.0
        edges = np.concatenate(([axis_bounds[0]], midpoints, [axis_bounds[1]]))

    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [(float(edges[run[0]]), float(edges[run[-1] + 1])) for run in runs]


def high_tortuosity_boxes(
    metrics_by_axis: dict[str, pd.DataFrame],
    mesh_bounds: np.ndarray,
    multiplier: float = 1.2,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, list[tuple[float, float]]],
    list[HighlightBox],
]:
    means, thresholds = axis_median_tortuosity_thresholds(metrics_by_axis, multiplier=multiplier)
    intervals_by_axis = {
        axis: contiguous_high_tortuosity_intervals(
            metrics_by_axis[axis],
            axis=axis,
            threshold=thresholds[axis],
            axis_bounds=(float(mesh_bounds[0, axis_index]), float(mesh_bounds[1, axis_index])),
        )
        for axis_index, axis in enumerate(("x", "y", "z"))
    }
    full_bounds: BoxBounds = (
        float(mesh_bounds[0, 0]),
        float(mesh_bounds[1, 0]),
        float(mesh_bounds[0, 1]),
        float(mesh_bounds[1, 1]),
        float(mesh_bounds[0, 2]),
        float(mesh_bounds[1, 2]),
    )
    boxes: list[HighlightBox] = []
    for axis_index, axis in enumerate(("x", "y", "z")):
        for interval in intervals_by_axis[axis]:
            bounds = list(full_bounds)
            bounds[axis_index * 2 : axis_index * 2 + 2] = interval
            boxes.append((axis, tuple(bounds)))
    return means, thresholds, intervals_by_axis, boxes


def skeleton_polydata(
    skel: sk.skeletonize.base.Skeleton,
    edge_to_branch: dict[tuple[int, int], int] | None = None,
) -> pv.PolyData:
    edges = np.asarray(skel.edges, dtype=np.int64)
    lines = np.column_stack((np.full(len(edges), 2), edges)).ravel()
    poly = pv.PolyData(np.asarray(skel.vertices, dtype=float), lines=lines)
    if edge_to_branch is not None:
        poly.cell_data["branch_id"] = np.array(
            [edge_to_branch.get(edge_key(int(a), int(b)), -1) for a, b in edges],
            dtype=float,
        )
    if "radius" in skel.swc.columns:
        nodes = skel.swc.set_index("node_id")
        poly.cell_data["radius_um"] = np.array(
            [np.nanmean([nodes.at[int(a), "radius"], nodes.at[int(b), "radius"]]) for a, b in edges],
            dtype=float,
        )
    return poly


def add_reference_geometry(plotter: pv.Plotter, mesh_data: pv.PolyData) -> None:
    plotter.add_axes(line_width=2)
    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
        font_size=8,
        color="#36454f",
        xtitle=f"x ({DISTANCE_UNIT})",
        ytitle=f"y ({DISTANCE_UNIT})",
        ztitle=f"z ({DISTANCE_UNIT})",
    )
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.18)
    plotter.reset_camera_clipping_range()


def add_skeleton(
    plotter: pv.Plotter,
    skel: sk.skeletonize.base.Skeleton,
    edge_to_branch: dict[tuple[int, int], int],
    tube_radius: float,
    colored: bool,
) -> None:
    lines = skeleton_polydata(skel, edge_to_branch=edge_to_branch)
    try:
        skeleton = lines.tube(radius=tube_radius)
    except Exception:
        skeleton = lines

    if colored and "radius_um" in skeleton.cell_data:
        plotter.add_mesh(
            skeleton,
            scalars="radius_um",
            cmap="turbo",
            scalar_bar_args={"title": f"radius ({DISTANCE_UNIT})"},
        )
    elif colored and "branch_id" in skeleton.cell_data:
        plotter.add_mesh(skeleton, scalars="branch_id", cmap="turbo", show_scalar_bar=False)
    else:
        plotter.add_mesh(skeleton, color="#d62728", line_width=4)


def pyvista_screenshot_available() -> bool:
    global _PYVISTA_SCREENSHOT_AVAILABLE
    if _PYVISTA_SCREENSHOT_AVAILABLE is not None:
        return _PYVISTA_SCREENSHOT_AVAILABLE

    code = (
        "import pyvista as pv; "
        "p=pv.Plotter(off_screen=True, window_size=(24,24)); "
        "p.add_mesh(pv.Sphere(theta_resolution=8, phi_resolution=8)); "
        "p.screenshot(return_img=True); "
        "p.close()"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=WORKSPACE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=12,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _PYVISTA_SCREENSHOT_AVAILABLE = False
    else:
        _PYVISTA_SCREENSHOT_AVAILABLE = result.returncode == 0
    return _PYVISTA_SCREENSHOT_AVAILABLE


def pyvista_screenshot(
    mesh: trimesh.Trimesh,
    skel: sk.skeletonize.base.Skeleton | None,
    edge_to_branch: dict[tuple[int, int], int],
    window_size: tuple[int, int] = (900, 650),
    highlight_boxes: list[HighlightBox] | None = None,
) -> np.ndarray:
    mesh_data = pyvista_mesh(mesh)
    extent = float(np.ptp(np.asarray(mesh.bounds), axis=0).max())
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    plotter.add_mesh(
        mesh_data,
        color="#c9c3b5",
        opacity=0.28 if skel is not None else 0.78,
        smooth_shading=True,
    )
    highlight_colors = {"x": "#b3541e", "y": "#34865f", "z": "#4169a1"}
    for axis, bounds in highlight_boxes or []:
        plotter.add_mesh(
            pv.Box(bounds=bounds),
            color=highlight_colors[axis],
            opacity=0.22,
            show_edges=True,
            edge_color=highlight_colors[axis],
            line_width=1.2,
        )
    if skel is not None:
        add_skeleton(plotter, skel, edge_to_branch, tube_radius=extent * 0.0015, colored=False)
    add_reference_geometry(plotter, mesh_data)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def set_axes_equal(ax: plt.Axes, bounds: np.ndarray) -> None:
    center = bounds.mean(axis=0)
    radius = float(np.ptp(bounds, axis=0).max() / 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def matplotlib_screenshot(
    mesh: trimesh.Trimesh,
    skel: sk.skeletonize.base.Skeleton | None,
    window_size: tuple[int, int] = (900, 650),
    max_faces: int = 20000,
    highlight_boxes: list[HighlightBox] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(7)
    face_ids = np.arange(len(mesh.faces))
    if len(face_ids) > max_faces:
        face_ids = np.sort(rng.choice(face_ids, size=max_faces, replace=False))

    fig = plt.figure(figsize=(window_size[0] / 120, window_size[1] / 120), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    tris = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[face_ids]]
    mesh_only = skel is None
    mesh_collection = Poly3DCollection(
        tris,
        facecolor="#b7aa8c",
        edgecolor="#dedbd2" if mesh_only else "none",
        linewidth=0.03 if mesh_only else 0.0,
        alpha=0.90 if mesh_only else 0.22,
    )
    ax.add_collection3d(mesh_collection)

    highlight_colors = {"x": "#b3541e", "y": "#34865f", "z": "#4169a1"}
    for axis, (x0, x1, y0, y1, z0, z1) in highlight_boxes or []:
        corners = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )
        faces = corners[
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
            ]
        ]
        box_collection = Poly3DCollection(
            faces,
            facecolor=highlight_colors[axis],
            edgecolor=highlight_colors[axis],
            linewidth=0.8,
            alpha=0.22,
        )
        ax.add_collection3d(box_collection)

    if skel is not None and len(skel.edges):
        vertices = np.asarray(skel.vertices, dtype=float)
        lines = vertices[np.asarray(skel.edges, dtype=int)]
        skeleton_collection = Line3DCollection(lines, colors="#d62728", linewidths=1.2)
        ax.add_collection3d(skeleton_collection)

    set_axes_equal(ax, np.asarray(mesh.bounds, dtype=float))
    ax.view_init(elev=20, azim=-55)
    ax.set_xlabel(f"x ({DISTANCE_UNIT})")
    ax.set_ylabel(f"y ({DISTANCE_UNIT})")
    ax.set_zlabel(f"z ({DISTANCE_UNIT})")
    ax.grid(True, color="#dddddd", linewidth=0.4)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def reference_screenshot(
    mesh: trimesh.Trimesh,
    skel: sk.skeletonize.base.Skeleton | None,
    edge_to_branch: dict[tuple[int, int], int],
    use_pyvista: bool,
    window_size: tuple[int, int] = (900, 650),
    highlight_boxes: list[HighlightBox] | None = None,
) -> np.ndarray:
    if use_pyvista:
        return pyvista_screenshot(
            mesh,
            skel=skel,
            edge_to_branch=edge_to_branch,
            window_size=window_size,
            highlight_boxes=highlight_boxes,
        )
    return matplotlib_screenshot(
        mesh,
        skel=skel,
        window_size=window_size,
        highlight_boxes=highlight_boxes,
    )


def save_summary_figure(
    metrics_by_axis: dict[str, pd.DataFrame],
    tortuosity_points_by_axis: dict[str, pd.DataFrame],
    mesh_image: np.ndarray,
    overlay_image: np.ndarray,
    high_tortuosity_image: np.ndarray,
    high_tortuosity_thresholds: dict[str, float],
    high_tortuosity_box_count: int,
    figure_path: Path,
) -> None:
    metrics = metrics_by_axis["x"]
    fig = plt.figure(figsize=(15, 14), constrained_layout=True)
    grid = fig.add_gridspec(4, 6, height_ratios=(1.05, 1.0, 1.0, 1.2))
    ax_mesh = fig.add_subplot(grid[0, :3])
    ax_overlay = fig.add_subplot(grid[0, 3:])
    ax_branches = fig.add_subplot(grid[1, :3])
    ax_area = fig.add_subplot(grid[1, 3:])
    ax_tortuosity_x = fig.add_subplot(grid[2, :2])
    ax_tortuosity_y = fig.add_subplot(grid[2, 2:4], sharey=ax_tortuosity_x)
    ax_tortuosity_z = fig.add_subplot(grid[2, 4:], sharey=ax_tortuosity_x)
    ax_high_tortuosity = fig.add_subplot(grid[3, :])

    for ax, image, title in (
        (ax_mesh, mesh_image, "Mesh"),
        (ax_overlay, overlay_image, "Mesh + Skeleton"),
    ):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()

    ax_branches.plot(metrics["x_position_um"], metrics["branch_crossings"], color="#2f6f9f", linewidth=1.8)
    ax_branches.set_title("Skeleton Crossings at YZ Planes")
    ax_branches.set_ylabel("crossings")

    ax_area.plot(metrics["x_position_um"], metrics["summed_lumen_area_um2"], color="#7b4ea3", linewidth=1.8)
    ax_area.set_title("Summed Circular Lumen Area at YZ Planes")
    ax_area.set_ylabel(f"summed lumen area ({AREA_UNIT})")

    plane_names = {"x": "YZ", "y": "XZ", "z": "XY"}
    colors = {"x": "#b3541e", "y": "#34865f", "z": "#4169a1"}
    tortuosity_axes = {
        "x": ax_tortuosity_x,
        "y": ax_tortuosity_y,
        "z": ax_tortuosity_z,
    }
    for axis, ax in tortuosity_axes.items():
        axis_metrics = metrics_by_axis[axis]
        tortuosity_points = tortuosity_points_by_axis[axis]
        position_column = f"{axis}_position_um"
        ax.fill_between(
            axis_metrics[position_column],
            axis_metrics["tortuosity_q25"],
            axis_metrics["tortuosity_q75"],
            color=colors[axis],
            alpha=0.2,
            linewidth=0,
            label="25th-75th percentile",
            zorder=1,
        )
        # ax.scatter(
        #     tortuosity_points[position_column],
        #     tortuosity_points["local_tortuosity"],
        #     color=colors[axis],
        #     alpha=0.4,
        #     edgecolors="none",
        #     s=8,
        #     rasterized=True,
        #     label="individual values",
        #     zorder=2,
        # )
        ax.plot(
            axis_metrics[position_column],
            axis_metrics["median_local_tortuosity"],
            color="#252525",
            linewidth=2.2,
            linestyle="-",
            marker=None,
            solid_capstyle="round",
            label="median",
            zorder=4,
        )
        ax.axhline(
            high_tortuosity_thresholds[axis],
            color=colors[axis],
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
            label="1.2 x axis mean",
            zorder=3,
        )
        ax.set_title(f"Median Local Tortuosity at {plane_names[axis]} Planes")
        ax.set_xlabel(f"distance along {axis} ({DISTANCE_UNIT})")
    ax_tortuosity_x.set_ylabel("local tortuosity")
    ax_tortuosity_y.legend(loc="upper right", frameon=False, fontsize=8)

    for ax in (ax_branches, ax_area, *tortuosity_axes.values()):
        ax.grid(True, color="#dddddd", linewidth=0.6)
    for ax in (ax_branches, ax_area):
        ax.set_xlabel(f"distance along x ({DISTANCE_UNIT})")

    ax_high_tortuosity.imshow(high_tortuosity_image)
    threshold_text = ", ".join(
        f"{plane_names[axis]}/{axis}: {high_tortuosity_thresholds[axis]:.3f}"
        for axis in ("x", "y", "z")
    )
    ax_high_tortuosity.set_title(
        "High-Tortuosity Interval Boxes: median > 1.2 x corresponding axis mean\n"
        f"Thresholds: {threshold_text}; {high_tortuosity_box_count} boxes"
    )
    ax_high_tortuosity.set_axis_off()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def show_viewer(
    mesh: trimesh.Trimesh,
    skel: sk.skeletonize.base.Skeleton,
    edge_to_branch: dict[tuple[int, int], int],
) -> None:
    mesh_data = pyvista_mesh(mesh)
    extent = float(np.ptp(np.asarray(mesh.bounds), axis=0).max())
    plotter = pv.Plotter(window_size=(1500, 1000))
    plotter.set_background("white")
    plotter.add_mesh(mesh_data, color="#c9c3b5", opacity=0.25, smooth_shading=True)
    add_skeleton(plotter, skel, edge_to_branch, tube_radius=extent * 0.0012, colored=True)
    add_reference_geometry(plotter, mesh_data)
    plotter.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=Path("100k_faces.cleaned.surf.stl"), type=Path)
    parser.add_argument("--figure", type=Path, help="Path for the static summary figure.")
    parser.add_argument("--csv", type=Path, help="Path for the YZ-plane metrics CSV.")
    parser.add_argument(
        "--planes",
        type=int,
        default=1000,
        help="Number of evenly spaced planes along each coordinate axis.",
    )
    parser.add_argument("--bins", type=int, help="Deprecated alias for --planes.")
    parser.add_argument("--waves", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--radius-method", choices=("ray", "knn"), default="ray")
    parser.add_argument("--n-rays", type=int, default=20)
    parser.add_argument(
        "--tortuosity-window-nodes",
        type=int,
        default=2,
        help="Degree-2 skeleton nodes to include on each side of a crossing edge for local tortuosity.",
    )
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip skeletor post-processing cleanup.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planes = args.bins if args.bins is not None else args.planes
    if planes < 1:
        raise ValueError("--planes must be at least 1")
    if args.waves < 1:
        raise ValueError("--waves must be at least 1")
    if args.step_size < 1:
        raise ValueError("--step-size must be at least 1")
    if args.n_rays < 1:
        raise ValueError("--n-rays must be at least 1")
    if args.tortuosity_window_nodes < 0:
        raise ValueError("--tortuosity-window-nodes must be at least 0")

    input_path = args.input
    figure_path = args.figure or default_output_path(input_path, ".skeleton_metrics.png")
    csv_path = args.csv or default_output_path(input_path, ".skeleton_metrics.csv")

    mesh = load_trimesh(input_path)
    print(f"Loaded {input_path}: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    skel = skeletonize_mesh(
        mesh,
        waves=args.waves,
        step_size=args.step_size,
        radius_method=args.radius_method,
        n_rays=args.n_rays,
        cleanup=not args.no_cleanup,
    )
    print(f"Skeleton: {len(skel.vertices):,} nodes, {len(skel.edges):,} edges")

    samples, edge_to_branch = branch_samples(skel, tortuosity_window_nodes=args.tortuosity_window_nodes)
    metrics_by_axis = {
        axis: plane_metrics(samples, mesh=mesh, planes=planes, axis=axis)
        for axis in ("x", "y", "z")
    }
    tortuosity_means, tortuosity_thresholds, intervals_by_axis, highlight_boxes = high_tortuosity_boxes(
        metrics_by_axis,
        mesh_bounds=np.asarray(mesh.bounds, dtype=float),
    )
    tortuosity_points_by_axis = {
        axis: plane_tortuosity_points(samples, metrics=metrics_by_axis[axis], axis=axis)
        for axis in ("x", "y", "z")
    }
    metrics = metrics_by_axis["x"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics[
        [
            "x_position_um",
            "branch_crossings",
            "mean_radius_um",
            "summed_lumen_area_um2",
            "median_local_tortuosity",
        ]
    ].to_csv(csv_path, index=False)
    print(f"Wrote metrics CSV: {csv_path}")

    use_pyvista_screenshots = pyvista_screenshot_available()
    if not use_pyvista_screenshots:
        print("PyVista off-screen screenshots are unavailable; using Matplotlib reference renders.")

    mesh_image = reference_screenshot(mesh, skel=None, edge_to_branch=edge_to_branch, use_pyvista=use_pyvista_screenshots)
    overlay_image = reference_screenshot(mesh, skel=skel, edge_to_branch=edge_to_branch, use_pyvista=use_pyvista_screenshots)
    high_tortuosity_image = reference_screenshot(
        mesh,
        skel=skel,
        edge_to_branch=edge_to_branch,
        use_pyvista=use_pyvista_screenshots,
        window_size=(1500, 520),
        highlight_boxes=highlight_boxes,
    )
    interval_counts = ", ".join(f"{axis}={len(intervals_by_axis[axis])}" for axis in ("x", "y", "z"))
    threshold_summary = ", ".join(
        f"{axis} mean={tortuosity_means[axis]:.4f}, threshold={tortuosity_thresholds[axis]:.4f}"
        for axis in ("x", "y", "z")
    )
    print(f"High-tortuosity {threshold_summary}; intervals: {interval_counts}; boxes: {len(highlight_boxes)}")
    save_summary_figure(
        metrics_by_axis,
        tortuosity_points_by_axis,
        mesh_image,
        overlay_image,
        high_tortuosity_image,
        tortuosity_thresholds,
        len(highlight_boxes),
        figure_path,
    )
    print(f"Wrote summary figure: {figure_path}")

    if args.show_viewer:
        show_viewer(mesh, skel, edge_to_branch)


if __name__ == "__main__":
    main()
