from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.skeletonize_stl_metrics import (
    contiguous_high_tortuosity_intervals,
    high_tortuosity_boxes,
    plane_metrics,
    plane_tortuosity_points,
)


@pytest.fixture
def crossing_sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "x0": 0.0,
                "x1": 4.0,
                "y0": 10.0,
                "y1": 14.0,
                "z0": 100.0,
                "z1": 104.0,
                "r0": 2.0,
                "r1": 4.0,
                "length": 7.0,
                "local_tortuosity": 1.25,
            }
        ]
    )


@pytest.mark.parametrize(
    ("axis", "expected_positions"),
    [
        ("x", [0.0, 2.0, 4.0]),
        ("y", [10.0, 12.0, 14.0]),
        ("z", [100.0, 102.0, 104.0]),
    ],
)
def test_plane_metrics_supports_each_coordinate_axis(
    crossing_sample: pd.DataFrame,
    axis: str,
    expected_positions: list[float],
) -> None:
    mesh = SimpleNamespace(bounds=np.array([[0.0, 10.0, 100.0], [4.0, 14.0, 104.0]]))

    metrics = plane_metrics(crossing_sample, mesh=mesh, planes=3, axis=axis)

    assert metrics[f"{axis}_position_um"].tolist() == expected_positions
    assert metrics["branch_crossings"].tolist() == [1, 1, 1]
    assert metrics["median_local_tortuosity"].tolist() == [1.25, 1.25, 1.25]
    assert metrics["tortuosity_q25"].tolist() == [1.25, 1.25, 1.25]
    assert metrics["tortuosity_q75"].tolist() == [1.25, 1.25, 1.25]

    points = plane_tortuosity_points(crossing_sample, metrics=metrics, axis=axis)
    assert points[f"{axis}_position_um"].tolist() == expected_positions
    assert points["local_tortuosity"].tolist() == [1.25, 1.25, 1.25]


def test_plane_metrics_rejects_unknown_axis(crossing_sample: pd.DataFrame) -> None:
    mesh = SimpleNamespace(bounds=np.zeros((2, 3)))

    with pytest.raises(ValueError, match="Unsupported plane-normal axis"):
        plane_metrics(crossing_sample, mesh=mesh, planes=3, axis="q")


def test_plane_metrics_calculates_tortuosity_distribution(crossing_sample: pd.DataFrame) -> None:
    samples = pd.concat(
        [crossing_sample.assign(local_tortuosity=value) for value in (1.0, 2.0, 3.0, 4.0)],
        ignore_index=True,
    )
    mesh = SimpleNamespace(bounds=np.array([[0.0, 10.0, 100.0], [4.0, 14.0, 104.0]]))

    metrics = plane_metrics(samples, mesh=mesh, planes=1, axis="x")
    points = plane_tortuosity_points(samples, metrics=metrics, axis="x")

    assert metrics.at[0, "median_local_tortuosity"] == 2.5
    assert metrics.at[0, "tortuosity_q25"] == 1.75
    assert metrics.at[0, "tortuosity_q75"] == 3.25
    assert points["local_tortuosity"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_contiguous_high_tortuosity_intervals_keeps_runs_separate() -> None:
    metrics = pd.DataFrame(
        {
            "x_position_um": [0.0, 1.0, 2.0, 3.0, 4.0],
            "median_local_tortuosity": [1.0, 2.0, 2.0, 1.0, 2.0],
        }
    )

    intervals = contiguous_high_tortuosity_intervals(
        metrics,
        axis="x",
        threshold=1.5,
        axis_bounds=(0.0, 4.0),
    )

    assert intervals == [(0.5, 2.5), (3.5, 4.0)]


def test_high_tortuosity_boxes_uses_per_axis_thresholds_and_slabs() -> None:
    metrics_by_axis = {
        "x": pd.DataFrame(
            {
                "x_position_um": [0.0, 1.0, 2.0, 3.0, 4.0],
                "median_local_tortuosity": [0.0, 2.0, 2.0, 0.0, 2.0],
            }
        ),
        "y": pd.DataFrame(
            {
                "y_position_um": [10.0, 11.0, 12.0, 13.0, 14.0],
                "median_local_tortuosity": [0.0, 2.0, 2.0, 0.0, 0.0],
            }
        ),
        "z": pd.DataFrame(
            {
                "z_position_um": [100.0, 101.0, 102.0, 103.0, 104.0],
                "median_local_tortuosity": [2.0, 0.0, 2.0, 0.0, 0.0],
            }
        ),
    }
    mesh_bounds = np.array([[0.0, 10.0, 100.0], [4.0, 14.0, 104.0]])

    means, thresholds, intervals_by_axis, boxes = high_tortuosity_boxes(
        metrics_by_axis,
        mesh_bounds,
    )

    assert means == pytest.approx({"x": 1.2, "y": 0.8, "z": 0.8})
    assert thresholds == pytest.approx({"x": 1.44, "y": 0.96, "z": 0.96})
    assert {axis: len(intervals) for axis, intervals in intervals_by_axis.items()} == {
        "x": 2,
        "y": 1,
        "z": 2,
    }
    assert len(boxes) == 5
    assert boxes[0] == ("x", (0.5, 2.5, 10.0, 14.0, 100.0, 104.0))
