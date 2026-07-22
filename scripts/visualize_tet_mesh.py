#!/usr/bin/env python3
"""Visualize the current vessel tetrahedral mesh with PyVista."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
CACHE_DIR = WORKSPACE / ".cache"
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import numpy as np
import pyvista as pv
import vtk


def parse_vector(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.replace(";", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values, e.g. 1,0,0")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("vector values must be numeric") from exc
    if np.linalg.norm(values) == 0:
        raise argparse.ArgumentTypeError("vector must be non-zero")
    return values


def extract_cell_type(grid: pv.UnstructuredGrid, cell_type: int, name: str) -> pv.UnstructuredGrid:
    indices = np.flatnonzero(grid.celltypes == cell_type)
    if indices.size == 0:
        raise ValueError(f"No {name} cells found in {grid}")
    return grid.extract_cells(indices)


def downsample_cells(grid: pv.UnstructuredGrid, max_cells: int, seed: int) -> tuple[pv.UnstructuredGrid, bool]:
    if max_cells <= 0 or grid.n_cells <= max_cells:
        return grid, False
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(grid.n_cells, size=max_cells, replace=False))
    return grid.extract_cells(indices), True


def center_origin(grid: pv.DataSet) -> tuple[float, float, float]:
    bounds = grid.bounds
    return (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )


def add_reference_geometry(plotter: pv.Plotter, grid: pv.DataSet) -> None:
    plotter.add_axes(line_width=2)
    plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
        font_size=9,
        color="#36454f",
    )


def quality_clim(measure: str) -> tuple[float, float] | None:
    if measure in {"scaled_jacobian", "min_sicn", "min_sige"}:
        return (0.0, 1.0)
    return None


def build_plot(args: argparse.Namespace) -> pv.Plotter:
    mesh = pv.read(args.input)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()

    wall = extract_cell_type(mesh, vtk.VTK_TRIANGLE, "wall triangle")
    volume = extract_cell_type(mesh, vtk.VTK_TETRA, "tetrahedron")
    origin = args.clip_origin or center_origin(volume)
    normal = args.clip_normal

    plotter = pv.Plotter(
        off_screen=args.off_screen or args.screenshot is not None,
        window_size=args.window_size,
    )
    plotter.set_background(args.background)

    wall_surface = wall.extract_surface(algorithm="dataset_surface")
    plotter.add_mesh(
        wall_surface,
        name="wall",
        color=args.wall_color,
        opacity=args.wall_opacity,
        smooth_shading=False,
    )

    sampled = False
    display_tets: pv.DataSet | None = None

    if args.mode == "overview":
        plotter.add_mesh(volume.outline(), color="#222222", line_width=2)

    elif args.mode == "cutaway":
        clipped = volume.clip(normal=normal, origin=origin, invert=args.invert_clip)
        display_tets, sampled = downsample_cells(clipped, args.max_tets, args.seed)
        plotter.add_mesh(
            display_tets,
            name="fluid_tets",
            color=args.tet_color,
            opacity=args.tet_opacity,
            show_edges=True,
            edge_color=args.edge_color,
            line_width=args.edge_width,
        )

    elif args.mode == "slice":
        cut = volume.slice(normal=normal, origin=origin)
        plotter.add_mesh(
            cut,
            name="tet_slice",
            color=args.tet_color,
            opacity=1.0,
            show_edges=True,
            edge_color=args.edge_color,
            line_width=args.edge_width,
        )

    elif args.mode == "quality":
        qualified = volume.cell_quality(args.quality_measure)
        clipped = qualified.clip(normal=normal, origin=origin, invert=args.invert_clip)
        display_tets, sampled = downsample_cells(clipped, args.max_tets, args.seed)
        plotter.add_mesh(
            display_tets,
            name="tet_quality",
            scalars=args.quality_measure,
            cmap=args.quality_cmap,
            clim=quality_clim(args.quality_measure),
            opacity=args.tet_opacity,
            show_edges=args.show_quality_edges,
            edge_color=args.edge_color,
            line_width=args.edge_width,
            scalar_bar_args={"title": args.quality_measure},
        )

    add_reference_geometry(plotter, mesh)
    plotter.camera_position = "iso"
    plotter.camera.zoom(args.zoom)

    text = (
        f"{args.input.name}\n"
        f"mode: {args.mode}\n"
        f"points: {mesh.n_points:,} | wall triangles: {wall.n_cells:,} | tets: {volume.n_cells:,}"
    )
    if sampled and display_tets is not None:
        text += f"\nrendered tets sampled to: {display_tets.n_cells:,}"
    plotter.add_text(text, position="upper_left", font_size=9, color="#111111")
    return plotter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("100k_faces.tet.vtu"))
    parser.add_argument(
        "--mode",
        choices=("overview", "cutaway", "slice", "quality"),
        default="cutaway",
        help="Visualization style. Use quality to color tets by a VTK cell-quality metric.",
    )
    parser.add_argument("--screenshot", type=Path, help="Write a screenshot instead of opening a window.")
    parser.add_argument("--off-screen", action="store_true", help="Render without opening an interactive window.")
    parser.add_argument("--window-size", type=int, nargs=2, default=(1600, 1100))
    parser.add_argument("--clip-normal", type=parse_vector, default=(1.0, 0.0, 0.0))
    parser.add_argument("--clip-origin", type=parse_vector)
    parser.add_argument("--invert-clip", action="store_true")
    parser.add_argument("--max-tets", type=int, default=80000, help="Maximum tets to render after clipping; 0 disables sampling.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wall-opacity", type=float, default=0.18)
    parser.add_argument("--tet-opacity", type=float, default=0.32)
    parser.add_argument("--wall-color", default="#d8d2c4")
    parser.add_argument("--tet-color", default="#5f9db7")
    parser.add_argument("--edge-color", default="#1f2933")
    parser.add_argument("--edge-width", type=float, default=0.35)
    parser.add_argument("--background", default="white")
    parser.add_argument("--zoom", type=float, default=1.25)
    parser.add_argument("--quality-measure", default="scaled_jacobian")
    parser.add_argument("--quality-cmap", default="viridis")
    parser.add_argument("--show-quality-edges", action="store_true")
    args = parser.parse_args()

    plotter = build_plot(args)
    if args.screenshot is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=str(args.screenshot), auto_close=True)
        print(f"Wrote screenshot: {args.screenshot}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()
