"""The VTU/PVD writer, validated against an independent reader.

The writer is hand-rolled against the VTK XML spec, so the load-bearing test is
the round trip: every file must parse in ``meshio`` with identical points,
line connectivity, and point data. On top sit structural checks for the time
series (frame files exist, the collection indexes them in order).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ddtwin.geometry import Sheet
from ddtwin.physics import SimConfig, simulate
from ddtwin.vtk_export import export_vtk, write_pvd, write_vtu

# meshio is the independent reader; skip the whole module if it is not installed.
meshio = pytest.importorskip("meshio")


def _small() -> tuple[Sheet, torch.Tensor]:
    sheet = Sheet.cantilever(rows=3, cols=5, dtype=torch.float64)
    density = torch.full((sheet.num_nodes,), 0.4, dtype=torch.float64)
    return sheet, density


def test_vtu_round_trips_through_meshio(tmp_path) -> None:
    sheet, density = _small()
    path = write_vtu(
        tmp_path / "snapshot.vtu",
        sheet.rest,
        sheet.edges,
        {"density": density, "rest": sheet.rest},
    )
    mesh = meshio.read(path)

    assert np.allclose(mesh.points, sheet.rest.numpy())
    (cells,) = mesh.cells
    assert cells.type == "line"
    assert np.array_equal(cells.data, sheet.edges.numpy())
    assert np.allclose(mesh.point_data["density"], density.numpy())
    assert np.allclose(mesh.point_data["rest"], sheet.rest.numpy())


def test_pvd_time_series(tmp_path) -> None:
    sheet, density = _small()
    res = simulate(sheet, density, SimConfig(frames=6))
    pvd = write_pvd(tmp_path, res.positions, sheet.edges, {"density": density}, rest=sheet.rest)

    assert pvd.name == "trajectory.pvd"
    text = pvd.read_text()
    frames = sorted(tmp_path.glob("trajectory_*.vtu"))
    assert len(frames) == 6
    for i, frame in enumerate(frames):
        assert frame.name in text
        assert f'timestep="{i}"' in text

    # First frame is the rest configuration: displacement exactly zero.
    first = meshio.read(frames[0])
    assert np.allclose(first.point_data["displacement"], 0.0)
    # Last frame is deformed: displacement nonzero and consistent with positions.
    last = meshio.read(frames[-1])
    expected = res.positions[-1].numpy() - sheet.rest.numpy()
    assert np.allclose(last.point_data["displacement"], expected)
    assert float(np.abs(expected).max()) > 1e-6


def test_export_vtk_convenience(tmp_path) -> None:
    sheet, density = _small()
    res = simulate(sheet, density, SimConfig(frames=4))
    pvd = export_vtk(tmp_path, sheet, res)
    assert pvd.exists()
    mesh = meshio.read(sorted(tmp_path.glob("*.vtu"))[0])
    assert "density" in mesh.point_data and "displacement" in mesh.point_data


def test_writer_validates_shapes(tmp_path) -> None:
    sheet, density = _small()
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        write_vtu(tmp_path / "bad.vtu", sheet.rest[:, :2], sheet.edges)
    with pytest.raises(ValueError, match="expected"):
        write_vtu(tmp_path / "bad.vtu", sheet.rest, sheet.edges, {"density": density[:-1]})
