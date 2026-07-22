#!/usr/bin/env python3
"""Conservative STL cleanup for vessel-surface tetrahedral meshing."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pymeshfix
import trimesh


def default_output_path(input_path: Path) -> Path:
    if input_path.name.endswith(".surf.stl"):
        return input_path.with_name(input_path.name.replace(".surf.stl", ".cleaned.surf.stl"))
    return input_path.with_name(f"{input_path.stem}.cleaned.stl")


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle meshes found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh from {path}, got {type(loaded).__name__}")
    return loaded


def conservative_cleanup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    finite_faces = np.isfinite(mesh.vertices[mesh.faces]).all(axis=(1, 2))
    mesh.update_faces(finite_faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-12))
    mesh.remove_unreferenced_vertices()
    return mesh


def edge_components(faces: np.ndarray) -> tuple[np.ndarray, list[int]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a = int(face[i])
            b = int(face[j])
            edge = (a, b) if a < b else (b, a)
            edge_to_faces[edge].append(face_index)

    adjacency: list[list[int]] = [[] for _ in range(len(faces))]
    for incident_faces in edge_to_faces.values():
        if len(incident_faces) < 2:
            continue
        for face_index in incident_faces:
            adjacency[face_index].extend(other for other in incident_faces if other != face_index)

    labels = np.full(len(faces), -1, dtype=np.int64)
    sizes: list[int] = []
    for start in range(len(faces)):
        if labels[start] != -1:
            continue
        label = len(sizes)
        queue: deque[int] = deque([start])
        labels[start] = label
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in adjacency[current]:
                if labels[neighbor] == -1:
                    labels[neighbor] = label
                    queue.append(neighbor)
        sizes.append(size)
    return labels, sizes


def keep_largest_component(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    labels, sizes = edge_components(mesh.faces)
    largest_label = int(np.argmax(sizes))
    keep = labels == largest_label
    kept_faces = int(keep.sum())
    removed_faces = int(len(mesh.faces) - kept_faces)

    largest = mesh.copy()
    largest.update_faces(keep)
    largest.remove_unreferenced_vertices()

    return largest, {
        "component_count": len(sizes),
        "component_face_counts_desc": sorted((int(size) for size in sizes), reverse=True),
        "kept_faces": kept_faces,
        "removed_faces": removed_faces,
    }


def edge_stats(faces: np.ndarray) -> dict[str, Any]:
    edge_counts: Counter[tuple[int, int]] = Counter()
    directed_counts: Counter[tuple[int, int]] = Counter()
    for face in faces:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a = int(face[i])
            b = int(face[j])
            edge = (a, b) if a < b else (b, a)
            edge_counts[edge] += 1
            directed_counts[(a, b)] += 1

    same_direction_conflicts = 0
    for a, b in edge_counts:
        if edge_counts[(a, b)] == 2 and (directed_counts[(a, b)] == 2 or directed_counts[(b, a)] == 2):
            same_direction_conflicts += 1

    return {
        "unique_edges": len(edge_counts),
        "boundary_edges_count1": sum(1 for count in edge_counts.values() if count == 1),
        "nonmanifold_edges_count_gt2": sum(1 for count in edge_counts.values() if count > 2),
        "max_edge_incidence": max(edge_counts.values()) if edge_counts else 0,
        "manifold_edges_with_same_direction_uses": same_direction_conflicts,
    }


def summarize_mesh(name: str, mesh: trimesh.Trimesh) -> dict[str, Any]:
    labels, sizes = edge_components(mesh.faces)
    stats = edge_stats(mesh.faces)
    bounds = mesh.bounds if len(mesh.vertices) else np.zeros((2, 3), dtype=float)
    sorted_sizes = sorted((int(size) for size in sizes), reverse=True)
    summary: dict[str, Any] = {
        "name": name,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_min": [float(x) for x in bounds[0]],
        "bounds_max": [float(x) for x in bounds[1]],
        "component_count": int(len(sizes)),
        "component_face_counts_desc_top20": sorted_sizes[:20],
        "component_face_counts_omitted": max(0, len(sorted_sizes) - 20),
        "euler_characteristic": int(len(mesh.vertices) - stats["unique_edges"] + len(mesh.faces)),
        "trimesh_is_watertight": bool(mesh.is_watertight),
        "surface_area": float(mesh.area),
        "volume": float(mesh.volume),
    }
    summary.update(stats)
    return summary


def repair_with_meshfix(mesh: trimesh.Trimesh, verbose: bool = False) -> trimesh.Trimesh:
    fixer = pymeshfix.MeshFix(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
        verbose=verbose,
    )
    fixer.repair(joincomp=False, remove_smallest_components=True)
    repaired = trimesh.Trimesh(
        vertices=np.asarray(fixer.points, dtype=np.float64),
        faces=np.asarray(fixer.faces, dtype=np.int64),
        process=False,
    )
    repaired = conservative_cleanup(repaired)
    repaired.fix_normals()
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="100k_faces.surf.stl", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verbose-meshfix", action="store_true")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or default_output_path(input_path)
    report_path = args.report or output_path.with_suffix(".report.json")

    raw = load_mesh(input_path)
    merged = conservative_cleanup(raw)
    largest, component_selection = keep_largest_component(merged)
    largest.fix_normals()
    repaired = repair_with_meshfix(largest, verbose=args.verbose_meshfix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(output_path, file_type="stl")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "steps": [
            "load STL without processing",
            "merge exact duplicate vertices and remove duplicate/degenerate faces",
            "keep largest edge-connected component",
            "run pymeshfix repair without joining components",
            "merge/validate repaired output and write binary STL",
        ],
        "component_selection": component_selection,
        "summaries": {
            "raw_loaded": summarize_mesh("raw_loaded", raw),
            "merged_for_topology": summarize_mesh("merged_for_topology", merged),
            "largest_component": summarize_mesh("largest_component", largest),
            "repaired_output": summarize_mesh("repaired_output", repaired),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    final = report["summaries"]["repaired_output"]
    print(f"Wrote cleaned STL: {output_path}")
    print(f"Wrote cleanup report: {report_path}")
    print(
        "Final topology: "
        f"{final['vertices']} vertices, {final['faces']} faces, "
        f"{final['component_count']} component(s), "
        f"{final['boundary_edges_count1']} boundary edge(s), "
        f"{final['nonmanifold_edges_count_gt2']} non-manifold edge(s)"
    )


if __name__ == "__main__":
    main()
