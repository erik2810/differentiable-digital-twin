"""Geometry and topology of the deformable sheet.

The digital twin is a rectangular mass-spring sheet discretised on a ``rows x cols``
grid. One edge is clamped (a cantilever); an external load is applied at the opposite
free edge. Springs come in three families:

    structural  axis-aligned neighbours          (carry stretch)
    shear       diagonal neighbours              (resist in-plane shear)
    bend        two-apart neighbours             (resist out-of-plane flexing)

Node ``(i, j)`` (row ``i``, column ``j``) maps to flat index ``i * cols + j``. Rest
positions lie in the ``x-z`` plane: ``x`` is the cantilever axis (clamp -> tip) and ``z``
is its height. The default load is in-plane (``-z`` at the tip), so the panel deforms
within its own plane -- the canonical 2D compliance-minimisation setup -- while the web
player extrudes the optimised density as height for a 3D relief of the load path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Spring family tags (stored per edge in ``Sheet.edge_kind``).
KIND_STRUCT = 0
KIND_SHEAR = 1
KIND_BEND = 2


@dataclass
class Sheet:
    """A discretised deformable sheet and its spring topology.

    All tensors live on the same device/dtype. ``rest`` are the undeformed node
    positions; everything else is connectivity that never changes during a sim.
    """

    rows: int
    cols: int
    rest: Tensor  # (N, 3) float  undeformed positions
    edges: Tensor  # (E, 2) long  spring endpoint indices
    rest_length: Tensor  # (E,)   float  undeformed spring lengths
    edge_kind: Tensor  # (E,)   long   one of KIND_*
    pinned: Tensor  # (N,)   bool   clamped (Dirichlet) nodes
    loaded: Tensor  # (N,)   bool   nodes where the external load is applied
    faces: Tensor  # (F, 3) long   triangles (two per grid cell) for rendering

    @property
    def num_nodes(self) -> int:
        return self.rows * self.cols

    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def device(self) -> torch.device:
        return self.rest.device

    def to(
        self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> Sheet:
        """Return a copy with float tensors moved to ``device``/``dtype``."""
        return Sheet(
            rows=self.rows,
            cols=self.cols,
            rest=self.rest.to(device=device, dtype=dtype),
            edges=self.edges.to(device=device),
            rest_length=self.rest_length.to(device=device, dtype=dtype),
            edge_kind=self.edge_kind.to(device=device),
            pinned=self.pinned.to(device=device),
            loaded=self.loaded.to(device=device),
            faces=self.faces.to(device=device),
        )

    @classmethod
    def cantilever(
        cls,
        rows: int = 12,
        cols: int = 20,
        spacing: float = 1.0,
        *,
        load_columns: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> Sheet:
        """Build a cantilever sheet clamped on the ``j = 0`` edge.

        Parameters
        ----------
        rows, cols
            Grid resolution. ``cols`` runs along the cantilever (clamp -> tip).
        spacing
            Rest distance between adjacent nodes.
        load_columns
            Number of rightmost columns that receive the external tip load.
        """
        if rows < 3 or cols < 3:
            raise ValueError("cantilever needs at least a 3x3 grid for bend springs")

        def idx(i: int, j: int) -> int:
            return i * cols + j

        # --- Rest positions: x = column (length), z = row (width, centred), y = 0.
        rest = torch.empty((rows * cols, 3), dtype=dtype, device=device)
        z0 = (rows - 1) / 2.0
        for i in range(rows):
            for j in range(cols):
                rest[idx(i, j)] = torch.tensor(
                    [j * spacing, 0.0, (i - z0) * spacing], dtype=dtype, device=device
                )

        # --- Springs.
        edge_list: list[tuple[int, int]] = []
        kind_list: list[int] = []

        def add(a: int, b: int, kind: int) -> None:
            edge_list.append((a, b))
            kind_list.append(kind)

        for i in range(rows):
            for j in range(cols):
                if j + 1 < cols:
                    add(idx(i, j), idx(i, j + 1), KIND_STRUCT)
                if i + 1 < rows:
                    add(idx(i, j), idx(i + 1, j), KIND_STRUCT)
                if i + 1 < rows and j + 1 < cols:
                    add(idx(i, j), idx(i + 1, j + 1), KIND_SHEAR)
                    add(idx(i + 1, j), idx(i, j + 1), KIND_SHEAR)
                if j + 2 < cols:
                    add(idx(i, j), idx(i, j + 2), KIND_BEND)
                if i + 2 < rows:
                    add(idx(i, j), idx(i + 2, j), KIND_BEND)

        edges = torch.tensor(edge_list, dtype=torch.long, device=device)
        edge_kind = torch.tensor(kind_list, dtype=torch.long, device=device)
        diff = rest[edges[:, 0]] - rest[edges[:, 1]]
        rest_length = diff.norm(dim=1)

        # --- Boundary conditions.
        pinned = torch.zeros(rows * cols, dtype=torch.bool, device=device)
        loaded = torch.zeros(rows * cols, dtype=torch.bool, device=device)
        for i in range(rows):
            pinned[idx(i, 0)] = True
            for j in range(cols - load_columns, cols):
                loaded[idx(i, j)] = True

        # --- Render faces (two CCW triangles per grid cell, normals facing +y).
        face_list: list[tuple[int, int, int]] = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)
                face_list.append((a, c, b))
                face_list.append((b, c, d))
        faces = torch.tensor(face_list, dtype=torch.long, device=device)

        return cls(
            rows=rows,
            cols=cols,
            rest=rest,
            edges=edges,
            rest_length=rest_length,
            edge_kind=edge_kind,
            pinned=pinned,
            loaded=loaded,
            faces=faces,
        )
