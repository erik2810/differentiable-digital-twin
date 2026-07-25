"""Export results to VTK files (VTU/PVD) for ParaView, PyVista, and friends.

The web player speaks the portfolio's own binary format; the scientific
visualisation world speaks VTK. This module writes the truss as a VTK
unstructured grid: nodes as points, members as ``VTK_LINE`` cells, and fields
(density, displacement) as point data. A settling trajectory becomes a ``.pvd``
collection, one ``.vtu`` per frame with a time stamp, which ParaView plays as an
animation out of the box.

The writer is hand-rolled against the VTK XML format (ASCII arrays, an
``UnstructuredGrid`` piece with connectivity/offsets/types, a ``Collection``
index for time series) rather than pulling in the VTK stack as a dependency.
The test suite round-trips every file through ``meshio`` as an independent
reader, so the output is verified against a spec-conform implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ddtwin.geometry import Sheet
from ddtwin.optimize import InverseDesignResult
from ddtwin.physics import SimResult

_VTK_LINE = 3  # VTK cell type id for a two-point line element


def _as_numpy(a: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _ascii(a: np.ndarray) -> str:
    """Flatten an array into the compact ASCII payload of a VTK DataArray."""
    return " ".join(f"{v:.9g}" for v in np.asarray(a, dtype=np.float64).ravel())


def _ascii_int(a: np.ndarray) -> str:
    return " ".join(str(int(v)) for v in np.asarray(a).ravel())


def write_vtu(
    path: str | Path,
    points: Tensor | np.ndarray,
    lines: Tensor | np.ndarray,
    point_data: dict[str, Tensor | np.ndarray] | None = None,
) -> Path:
    """Write one snapshot as a VTK unstructured grid (``.vtu``).

    Args:
        path: Output file; parent directories are created.
        points: Node positions, shape ``(N, 3)``.
        lines: Line cells (truss members / springs), shape ``(E, 2)``,
            integer node indices.
        point_data: Optional per-node fields, each of shape ``(N,)`` (scalar)
            or ``(N, 3)`` (vector).

    Returns:
        The written path.
    """
    pts = _as_numpy(points)
    con = _as_numpy(lines)
    n, e = pts.shape[0], con.shape[0]
    if pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")

    arrays: list[str] = []
    if point_data:
        for name, field in point_data.items():
            arr = _as_numpy(field)
            comps = 1 if arr.ndim == 1 else arr.shape[1]
            if arr.shape[0] != n:
                raise ValueError(f"point_data '{name}' has {arr.shape[0]} entries, expected {n}")
            arrays.append(
                f'<DataArray type="Float64" Name="{name}" '
                f'NumberOfComponents="{comps}" format="ascii">{_ascii(arr)}</DataArray>'
            )
    point_data_block = f"<PointData>{''.join(arrays)}</PointData>" if arrays else ""

    offsets = np.arange(1, e + 1) * 2
    types = np.full(e, _VTK_LINE)
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        "<UnstructuredGrid>\n"
        f'<Piece NumberOfPoints="{n}" NumberOfCells="{e}">\n'
        "<Points>"
        f'<DataArray type="Float64" NumberOfComponents="3" format="ascii">{_ascii(pts)}</DataArray>'
        "</Points>\n"
        "<Cells>"
        f'<DataArray type="Int64" Name="connectivity" format="ascii">{_ascii_int(con)}</DataArray>'
        f'<DataArray type="Int64" Name="offsets" format="ascii">{_ascii_int(offsets)}</DataArray>'
        f'<DataArray type="UInt8" Name="types" format="ascii">{_ascii_int(types)}</DataArray>'
        "</Cells>\n"
        f"{point_data_block}\n"
        "</Piece>\n"
        "</UnstructuredGrid>\n"
        "</VTKFile>\n"
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)
    return out


def write_pvd(
    out_dir: str | Path,
    positions: Tensor | np.ndarray,
    lines: Tensor | np.ndarray,
    point_data: dict[str, Tensor | np.ndarray] | None = None,
    *,
    name: str = "trajectory",
    rest: Tensor | np.ndarray | None = None,
) -> Path:
    """Write a trajectory as a ``.pvd`` time series of ``.vtu`` frames.

    Args:
        out_dir: Target directory (created if needed).
        positions: Node trajectory, shape ``(F, N, 3)``.
        lines: Line cells shared by all frames, shape ``(E, 2)``.
        point_data: Static per-node fields written into every frame
            (e.g. the optimised density).
        name: Base name: frames become ``<name>_0000.vtu`` and the index
            ``<name>.pvd``.
        rest: Optional rest positions ``(N, 3)``; when given, each frame also
            carries the per-frame ``displacement`` vector ``positions[f] - rest``.

    Returns:
        The path of the ``.pvd`` collection file.
    """
    traj = _as_numpy(positions)
    if traj.ndim != 3 or traj.shape[2] != 3:
        raise ValueError(f"positions must be (F, N, 3), got {traj.shape}")
    rest_np = _as_numpy(rest) if rest is not None else None

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    for f in range(traj.shape[0]):
        frame_data: dict[str, Tensor | np.ndarray] = dict(point_data or {})
        if rest_np is not None:
            frame_data["displacement"] = traj[f] - rest_np
        fname = f"{name}_{f:04d}.vtu"
        write_vtu(out / fname, traj[f], lines, frame_data)
        entries.append(f'<DataSet timestep="{f}" group="" part="0" file="{fname}"/>')

    pvd = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "<Collection>\n" + "\n".join(entries) + "\n</Collection>\n</VTKFile>\n"
    )
    pvd_path = out / f"{name}.pvd"
    pvd_path.write_text(pvd)
    return pvd_path


def export_vtk(out_dir: str | Path, sheet: Sheet, result: SimResult | InverseDesignResult) -> Path:
    """Export a simulation or optimisation result as a ParaView-ready time series.

    Writes the settling trajectory with the density field and per-frame
    displacement vectors; returns the ``.pvd`` path. Open it in ParaView (or
    ``pyvista.read``) and press play.
    """
    sim = result.final if isinstance(result, InverseDesignResult) else result
    return write_pvd(
        out_dir,
        sim.positions,
        sheet.edges,
        {"density": sim.density},
        rest=sheet.rest,
    )
