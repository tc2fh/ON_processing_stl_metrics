#!/usr/bin/env python3
"""Generate a tetrahedral volume mesh from the cleaned vessel STL with Gmsh."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gmsh
import meshio
import numpy as np
import trimesh


def default_path(input_path: Path, suffix: str) -> Path:
    if input_path.name.endswith(".cleaned.surf.stl"):
        return input_path.with_name(input_path.name.replace(".cleaned.surf.stl", suffix))
    if input_path.name.endswith(".surf.stl"):
        return input_path.with_name(input_path.name.replace(".surf.stl", suffix))
    return input_path.with_name(f"{input_path.stem}{suffix}")


def surface_edge_lengths(path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)


def choose_default_sizes(path: Path) -> dict[str, float]:
    lengths = surface_edge_lengths(path)
    p01, p10, p50, p90 = np.percentile(lengths, [1, 10, 50, 90])
    return {
        "surface_edge_p01": float(p01),
        "surface_edge_p10": float(p10),
        "surface_edge_p50": float(p50),
        "surface_edge_p90": float(p90),
        "mesh_size_min": float(max(p01, p50 / 4.0)),
        "mesh_size_max": float(p50 * 3.0),
    }


def set_gmsh_options(mesh_size_min: float, mesh_size_max: float, algorithm3d: int, verbose: bool) -> None:
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
    gmsh.option.setNumber("Mesh.Algorithm3D", algorithm3d)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.SaveAll", 0)


def cell_counts(mesh: meshio.Mesh) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in mesh.cells:
        counts[block.type] = counts.get(block.type, 0) + int(len(block.data))
    return counts


def tetra_edge_ratios(mesh: meshio.Mesh) -> dict[str, float] | None:
    tets = [block.data for block in mesh.cells if block.type == "tetra"]
    if not tets:
        return None
    tetra = np.vstack(tets)
    points = np.asarray(mesh.points)
    tet_points = points[tetra]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    lengths = np.stack(
        [np.linalg.norm(tet_points[:, a] - tet_points[:, b], axis=1) for a, b in edge_pairs],
        axis=1,
    )
    ratios = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), np.finfo(float).eps)
    volumes = np.abs(
        np.einsum(
            "ij,ij->i",
            tet_points[:, 1] - tet_points[:, 0],
            np.cross(tet_points[:, 2] - tet_points[:, 0], tet_points[:, 3] - tet_points[:, 0]),
        )
    ) / 6.0
    return {
        "edge_ratio_p50": float(np.percentile(ratios, 50)),
        "edge_ratio_p90": float(np.percentile(ratios, 90)),
        "edge_ratio_p99": float(np.percentile(ratios, 99)),
        "edge_ratio_max": float(ratios.max()),
        "tet_volume_sum": float(volumes.sum()),
        "tet_volume_min": float(volumes.min()),
        "tet_volume_p50": float(np.percentile(volumes, 50)),
    }


def build_mesh(
    input_path: Path,
    msh_path: Path,
    mesh_size_min: float,
    mesh_size_max: float,
    classification_angle: float,
    curve_angle: float,
    force_parametrizable_patches: bool,
    reparametrize: bool,
    algorithm3d: int,
    verbose: bool,
) -> dict[str, Any]:
    gmsh.initialize()
    try:
        set_gmsh_options(mesh_size_min, mesh_size_max, algorithm3d, verbose)
        gmsh.model.add("vessel_lumen")
        gmsh.merge(str(input_path))

        if reparametrize:
            gmsh.model.mesh.classifySurfaces(
                classification_angle * math.pi / 180.0,
                True,
                force_parametrizable_patches,
                curve_angle * math.pi / 180.0,
            )
            gmsh.model.mesh.createGeometry()

        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            raise RuntimeError("Gmsh did not create any surfaces from the STL")
        surface_tags = [tag for _, tag in surfaces]

        loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
        volume = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()

        wall_group = gmsh.model.addPhysicalGroup(2, surface_tags, tag=1)
        fluid_group = gmsh.model.addPhysicalGroup(3, [volume], tag=1)
        gmsh.model.setPhysicalName(2, wall_group, "wall")
        gmsh.model.setPhysicalName(3, fluid_group, "fluid")

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        gmsh.write(str(msh_path))

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        element_counts = {}
        for dim in (2, 3):
            element_types, element_tags, _ = gmsh.model.mesh.getElements(dim)
            for element_type, tags in zip(element_types, element_tags):
                name, _, _, _, _, _ = gmsh.model.mesh.getElementProperties(element_type)
                element_counts[name] = element_counts.get(name, 0) + int(len(tags))

        return {
            "gmsh_node_count": int(len(node_tags)),
            "gmsh_element_counts": element_counts,
            "surface_patch_count": int(len(surface_tags)),
            "surface_tags": surface_tags,
            "volume_tag": int(volume),
            "mode": "reparametrize" if reparametrize else "discrete_surface",
        }
    finally:
        gmsh.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="100k_faces.cleaned.surf.stl", type=Path)
    parser.add_argument("--msh", type=Path)
    parser.add_argument("--vtu", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--mesh-size-min", type=float)
    parser.add_argument("--mesh-size-max", type=float)
    parser.add_argument("--classification-angle", type=float, default=40.0)
    parser.add_argument("--curve-angle", type=float, default=180.0)
    parser.add_argument(
        "--reparametrize",
        action="store_true",
        help="Classify/reparametrize STL patches before meshing. Default keeps the STL as a discrete surface.",
    )
    parser.add_argument(
        "--no-force-parametrizable-patches",
        action="store_true",
        help="Disable Gmsh patch splitting for parametrization.",
    )
    parser.add_argument("--algorithm3d", type=int, default=1)
    parser.add_argument("--verbose-gmsh", action="store_true")
    args = parser.parse_args()

    input_path = args.input
    msh_path = args.msh or default_path(input_path, ".tet.msh")
    vtu_path = args.vtu or default_path(input_path, ".tet.vtu")
    report_path = args.report or default_path(input_path, ".tet.report.json")

    defaults = choose_default_sizes(input_path)
    mesh_size_min = args.mesh_size_min if args.mesh_size_min is not None else defaults["mesh_size_min"]
    mesh_size_max = args.mesh_size_max if args.mesh_size_max is not None else defaults["mesh_size_max"]

    msh_path.parent.mkdir(parents=True, exist_ok=True)
    vtu_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    gmsh_summary = build_mesh(
        input_path=input_path,
        msh_path=msh_path,
        mesh_size_min=mesh_size_min,
        mesh_size_max=mesh_size_max,
        classification_angle=args.classification_angle,
        curve_angle=args.curve_angle,
        force_parametrizable_patches=not args.no_force_parametrizable_patches,
        reparametrize=args.reparametrize,
        algorithm3d=args.algorithm3d,
        verbose=args.verbose_gmsh,
    )

    volume_mesh = meshio.read(msh_path)
    meshio.write(vtu_path, volume_mesh)

    counts = cell_counts(volume_mesh)
    quality = tetra_edge_ratios(volume_mesh)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "msh": str(msh_path),
        "vtu": str(vtu_path),
        "mesh_size": {
            "min": mesh_size_min,
            "max": mesh_size_max,
            "chosen_from_surface_defaults": defaults,
        },
        "gmsh": {
            "classification_angle_degrees": args.classification_angle,
            "curve_angle_degrees": args.curve_angle,
            "force_parametrizable_patches": not args.no_force_parametrizable_patches,
            "reparametrize": args.reparametrize,
            "algorithm3d": args.algorithm3d,
            **gmsh_summary,
        },
        "meshio": {
            "point_count": int(len(volume_mesh.points)),
            "cell_counts": counts,
            "tetra_quality": quality,
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote MSH: {msh_path}")
    print(f"Wrote VTU: {vtu_path}")
    print(f"Wrote report: {report_path}")
    print(f"Cells: {counts}")
    if quality:
        print(
            "Tet edge ratio: "
            f"p50={quality['edge_ratio_p50']:.3g}, "
            f"p90={quality['edge_ratio_p90']:.3g}, "
            f"p99={quality['edge_ratio_p99']:.3g}, "
            f"max={quality['edge_ratio_max']:.3g}"
        )


if __name__ == "__main__":
    main()
