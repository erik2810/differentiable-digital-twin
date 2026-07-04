"""Differentiable static structural solver (linear-elastic truss FEM).

The sheet is modelled as a 2D in-plane elastic truss: every spring is a two-force member
whose axial stiffness is set by the local material density through a SIMP interpolation. We
assemble the global stiffness matrix ``K(rho)`` and solve the linear system

    K(rho) u = f

for the nodal displacement ``u`` under the applied load ``f``. The design objective is the
**compliance** ``C = f . u`` (the work done by the load = strain energy stored); minimising it
is equivalent to maximising stiffness. This is the canonical topology-optimisation problem and
is exactly how production structural optimisation is posed.

Why it is differentiable
------------------------
``torch.linalg.solve`` is differentiable, so ``dC/drho`` is obtained by reverse-mode autodiff
through the solve in a *single* backward pass -- cost independent of the number of design
variables. (Equivalently, this is the classic self-adjoint compliance sensitivity
``dC/dx_e = -u_e^T (dK_e/dx_e) u_e`` that adjoint topology optimisation derives by hand; here we
let autodiff compute it.)

The sheet lies in the ``x-z`` plane, so out-of-plane (``y``) DOFs carry no stiffness and are
constrained -- a standard plane reduction. For visualisation, :func:`simulate` wraps the static
solve and synthesises a load-application trajectory (the structure relaxing to ``u``) for the
web player.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ddtwin.geometry import KIND_BEND, KIND_SHEAR, Sheet


@dataclass
class SimConfig:
    """Material interpolation, load, and animation parameters."""

    load: tuple[float, float, float] = (0.0, 0.0, -1.0)  # total load, split over the loaded nodes

    # SIMP stiffness interpolation k(rho) = k_min + (k_max - k_min) * rho**penal
    k_min: float = 1.0
    k_max: float = 80.0
    penal: float = 3.0

    # Per-family axial-stiffness multipliers (diagonal/bend members are softer than structural).
    shear_scale: float = 0.9
    bend_scale: float = 0.5

    # Visualisation only: frames in the synthesised load-application animation and its liveliness.
    frames: int = 120
    overshoot: float = 0.12  # underdamped overshoot of the settling envelope (0 = pure ramp)


@dataclass
class StaticResult:
    """Result of a single static solve."""

    u: Tensor  # (N, 3) nodal displacement
    compliance: Tensor  # scalar  C = f . u
    rest: Tensor  # (N, 3)
    density: Tensor  # (N,)
    loaded: Tensor  # (N,) bool

    def tip_deflection(self) -> Tensor:
        """Mean displacement magnitude of the loaded nodes (an intuitive stiffness proxy)."""
        return self.u[self.loaded].norm(dim=1).mean()

    def max_deflection(self) -> Tensor:
        return self.u.norm(dim=1).max()


@dataclass
class SimResult:
    """Static solve plus a synthesised animation trajectory for the player."""

    positions: Tensor  # (frames, N, 3) rest + envelope(t) * u
    static: StaticResult

    @property
    def density(self) -> Tensor:
        return self.static.density

    def tip_deflection(self) -> Tensor:
        return self.static.tip_deflection()

    def compliance(self) -> Tensor:
        return self.static.compliance


def _free_dofs(sheet: Sheet) -> Tensor:
    """Boolean mask over the ``3N`` DOFs: free iff the node is unpinned and the axis is in-plane."""
    in_plane = torch.ones((sheet.num_nodes, 3), dtype=torch.bool, device=sheet.device)
    in_plane[:, 1] = False  # constrain out-of-plane (y)
    free = in_plane & ~sheet.pinned.unsqueeze(1)
    return free.reshape(-1)


def _edge_stiffness(sheet: Sheet, density: Tensor, cfg: SimConfig) -> Tensor:
    """SIMP axial stiffness per member, scaled by spring family. Shape ``(E,)``."""
    rho_edge = 0.5 * (density[sheet.edges[:, 0]] + density[sheet.edges[:, 1]])
    k = cfg.k_min + (cfg.k_max - cfg.k_min) * rho_edge.clamp(0.0, 1.0) ** cfg.penal
    scale = torch.ones_like(k)
    scale = torch.where(sheet.edge_kind == KIND_SHEAR, scale * cfg.shear_scale, scale)
    scale = torch.where(sheet.edge_kind == KIND_BEND, scale * cfg.bend_scale, scale)
    return k * scale


def _assemble_stiffness(sheet: Sheet, k: Tensor) -> Tensor:
    """Assemble the global stiffness matrix ``K`` (``3N x 3N``) from truss members.

    Each member ``e`` from ``a`` to ``b`` with unit axis ``d`` (at rest) contributes the
    rank-1 block ``k_e d d^T`` to the ``(a,a)`` and ``(b,b)`` diagonals and its negative to
    the ``(a,b)`` / ``(b,a)`` off-diagonals -- the standard two-force-member element.
    """
    n3 = 3 * sheet.num_nodes
    dtype, device = sheet.rest.dtype, sheet.device
    diff = sheet.rest[sheet.edges[:, 1]] - sheet.rest[sheet.edges[:, 0]]
    axis = diff / diff.norm(dim=1, keepdim=True).clamp_min(1e-12)  # (E, 3)
    block = k[:, None, None] * (axis[:, :, None] * axis[:, None, :])  # (E, 3, 3)

    a = sheet.edges[:, 0]
    b = sheet.edges[:, 1]
    rng = torch.arange(3, device=device)
    ar = 3 * a[:, None] + rng[None, :]  # (E, 3) global row indices for node a
    br = 3 * b[:, None] + rng[None, :]

    k_global = torch.zeros(n3 * n3, dtype=dtype, device=device)

    def scatter(rows: Tensor, cols: Tensor, values: Tensor, sign: float) -> None:
        flat = (rows[:, :, None] * n3 + cols[:, None, :]).reshape(-1)
        k_global.index_add_(0, flat, sign * values.reshape(-1))

    scatter(ar, ar, block, 1.0)
    scatter(br, br, block, 1.0)
    scatter(ar, br, block, -1.0)
    scatter(br, ar, block, -1.0)
    return k_global.reshape(n3, n3)


def solve_static(sheet: Sheet, density: Tensor, cfg: SimConfig | None = None) -> StaticResult:
    """Solve ``K(rho) u = f`` and return the displacement and compliance.

    Args:
        sheet: Geometry and topology; supplies the ``N`` nodes and the edge list.
        density: Per-node material density, shape ``(N,)``, values in ``(0, 1)``.
            Mapped to axial stiffness through the SIMP interpolation in ``cfg``.
        cfg: Material, load, and animation parameters; defaults to ``SimConfig()``.

    Returns:
        A :class:`StaticResult` with displacement ``u`` of shape ``(N, 3)`` (pinned
        and out-of-plane ``y`` DOFs are zero by construction) and the scalar
        compliance ``C = f . u``.

    Differentiating ``result.compliance`` w.r.t. ``density`` yields the full design gradient.
    """
    cfg = cfg or SimConfig()
    dtype, device = sheet.rest.dtype, sheet.device
    density = density.to(dtype=dtype, device=device)

    k = _edge_stiffness(sheet, density, cfg)
    k_global = _assemble_stiffness(sheet, k)

    # External load: a total load split over the loaded nodes.
    f = torch.zeros((sheet.num_nodes, 3), dtype=dtype, device=device)
    n_loaded = int(sheet.loaded.sum())
    if n_loaded > 0:
        load_vec = torch.tensor(cfg.load, dtype=dtype, device=device) / n_loaded
        f = torch.where(sheet.loaded.unsqueeze(1), load_vec, f)
    f = f.reshape(-1)

    free = _free_dofs(sheet)
    k_ff = k_global[free][:, free]
    f_f = f[free]
    u_f = torch.linalg.solve(k_ff, f_f)

    u = torch.zeros(3 * sheet.num_nodes, dtype=dtype, device=device)
    u = u.index_copy(0, free.nonzero(as_tuple=True)[0], u_f)
    compliance = f_f @ u_f
    return StaticResult(
        u=u.reshape(sheet.num_nodes, 3),
        compliance=compliance,
        rest=sheet.rest,
        density=density,
        loaded=sheet.loaded,
    )


def _settle_envelope(cfg: SimConfig, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Underdamped step response in ``[0, 1]``: the load is applied and the structure settles."""
    t = torch.linspace(0.0, 1.0, cfg.frames, dtype=dtype, device=device)
    if cfg.overshoot <= 0.0:
        return 1.0 - torch.exp(-5.0 * t)
    zeta = max(min(1.0 - cfg.overshoot, 0.999), 0.05)
    wd = torch.pi * 2.2
    decay = torch.exp(-zeta * 6.0 * t)
    env = 1.0 - decay * (torch.cos(wd * t) + (zeta / (1 - zeta**2) ** 0.5) * torch.sin(wd * t))
    return env.clamp_min(0.0)


def simulate(sheet: Sheet, density: Tensor, cfg: SimConfig | None = None) -> SimResult:
    """Static solve plus a synthesised settling animation (``rest + envelope(t) * u``).

    Returns a :class:`SimResult` whose ``positions`` have shape ``(frames, N, 3)``,
    starting at the rest configuration and settling into the deformed one.
    """
    cfg = cfg or SimConfig()
    static = solve_static(sheet, density, cfg)
    env = _settle_envelope(cfg, sheet.rest.dtype, sheet.device)  # (F,)
    positions = sheet.rest.unsqueeze(0) + env[:, None, None] * static.u.unsqueeze(0)
    return SimResult(positions=positions, static=static)
