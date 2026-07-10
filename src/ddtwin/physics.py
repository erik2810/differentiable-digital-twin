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

    # Solver. The dense path assembles the full (3N, 3N) stiffness matrix, whose
    # O(N^2) memory is what caps the mesh resolution. The sparse path never
    # assembles it: a matrix-free operator applies K in O(E) and a conjugate-
    # gradient solve replaces the factorisation, so memory grows O(N). Gradients
    # flow through a hand-written adjoint (one extra CG solve in backward),
    # verified against finite differences in the test suite.
    sparse: bool = False
    cg_tol: float | None = None  # relative residual; default 1e-12 (f64) / 1e-6 (f32)
    cg_max_iters: int | None = None  # default: number of free DOFs


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


def _edge_axes(sheet: Sheet) -> Tensor:
    """Unit rest axis of every member, shape ``(E, 3)``."""
    diff = sheet.rest[sheet.edges[:, 1]] - sheet.rest[sheet.edges[:, 0]]
    return diff / diff.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _assemble_stiffness(sheet: Sheet, k: Tensor) -> Tensor:
    """Assemble the global stiffness matrix ``K`` (``3N x 3N``) from truss members.

    Each member ``e`` from ``a`` to ``b`` with unit axis ``d`` (at rest) contributes the
    rank-1 block ``k_e d d^T`` to the ``(a,a)`` and ``(b,b)`` diagonals and its negative to
    the ``(a,b)`` / ``(b,a)`` off-diagonals -- the standard two-force-member element.
    """
    n3 = 3 * sheet.num_nodes
    dtype, device = sheet.rest.dtype, sheet.device
    axis = _edge_axes(sheet)  # (E, 3)
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


# ---------------------------------------------------------------------------
# Sparse (matrix-free) static solve
# ---------------------------------------------------------------------------


def _apply_stiffness(
    u: Tensor, k: Tensor, a: Tensor, b: Tensor, axes: Tensor, free: Tensor
) -> Tensor:
    """Apply the constrained stiffness operator ``P K P + (I - P)`` in O(E).

    ``K u`` is accumulated member by member: the axial stretch rate of edge ``e``
    is ``s_e = d_e . (u_a - u_b)``, and the member pushes ``k_e s_e d_e`` on node
    ``a`` and its negative on node ``b``. ``P`` projects onto the free DOFs; the
    identity on fixed DOFs keeps the operator symmetric positive definite over the
    whole ``3N`` space, so CG can run unmasked.

    Args:
        u: DOF vector, shape ``(3N,)``.
        k: Member stiffness, shape ``(E,)``.
        a, b: Member endpoints, shape ``(E,)`` each.
        axes: Member unit axes, shape ``(E, 3)``.
        free: Free-DOF mask, shape ``(3N,)`` bool.

    Returns:
        The operator applied to ``u``, shape ``(3N,)``.
    """
    u_masked = torch.where(free, u, torch.zeros((), dtype=u.dtype, device=u.device))
    u3 = u_masked.reshape(-1, 3)
    stretch = (axes * (u3[a] - u3[b])).sum(dim=1)  # (E,)
    w = (k * stretch)[:, None] * axes  # (E, 3) axial force per member
    y3 = torch.zeros_like(u3)
    y3.index_add_(0, a, w)
    y3.index_add_(0, b, -w)
    return torch.where(free, y3.reshape(-1), u)


def _stiffness_diagonal(k: Tensor, a: Tensor, b: Tensor, axes: Tensor, free: Tensor) -> Tensor:
    """Diagonal of the constrained operator (Jacobi preconditioner), shape ``(3N,)``."""
    contrib = k[:, None] * axes * axes  # (E, 3) diagonal block entries per member
    d3 = torch.zeros((free.numel() // 3, 3), dtype=k.dtype, device=k.device)
    d3.index_add_(0, a, contrib)
    d3.index_add_(0, b, contrib)
    diag = d3.reshape(-1)
    one = torch.ones((), dtype=k.dtype, device=k.device)
    return torch.where(free, diag.clamp_min(1e-30), one)


def _cg_solve(
    k: Tensor,
    rhs: Tensor,
    a: Tensor,
    b: Tensor,
    axes: Tensor,
    free: Tensor,
    tol: float,
    max_iters: int,
) -> Tensor:
    """Jacobi-preconditioned conjugate gradients for the constrained operator.

    Args:
        k: Member stiffness, shape ``(E,)``.
        rhs: Right-hand side, shape ``(3N,)`` (zero on fixed DOFs).
        tol: Relative residual target ``|r| <= tol * |rhs|``.
        max_iters: Iteration cap.

    Returns:
        Solution, shape ``(3N,)``, zero on fixed DOFs.
    """
    diag = _stiffness_diagonal(k, a, b, axes, free)
    x = torch.zeros_like(rhs)
    rhs_norm = rhs.norm()
    if float(rhs_norm) == 0.0:
        return x  # zero load (or zero grad_output in the adjoint): the solution is zero
    r = rhs.clone()
    z = r / diag
    p = z.clone()
    rz = r @ z
    target = tol * rhs_norm
    for _ in range(max_iters):
        ap = _apply_stiffness(p, k, a, b, axes, free)
        alpha = rz / (p @ ap)
        x = x + alpha * p
        r = r - alpha * ap
        if r.norm() <= target:
            break
        z = r / diag
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


class _SparseStaticSolve(torch.autograd.Function):
    """Differentiable matrix-free solve of ``K(k) u = f`` via the adjoint method.

    The dense path lets autodiff differentiate through the factorisation, which
    requires the (3N, 3N) matrix to exist. Here nothing is assembled, so the
    adjoint is written out by hand: with ``L`` any scalar downstream of ``u``,

        K lambda = dL/du            (one extra CG solve; K is symmetric)
        dL/dk_e  = -(d_e . (lambda_a - lambda_b)) (d_e . (u_a - u_b))
        dL/df    = lambda

    which is the classic compliance-style sensitivity, generalised to any loss.
    Verified against finite differences by ``gradcheck`` in the test suite.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        k: Tensor,
        f: Tensor,
        a: Tensor,
        b: Tensor,
        axes: Tensor,
        free: Tensor,
        tol: float,
        max_iters: int,
    ) -> Tensor:
        with torch.no_grad():
            rhs = torch.where(free, f, torch.zeros((), dtype=f.dtype, device=f.device))
            u = _cg_solve(k, rhs, a, b, axes, free, tol, max_iters)
        ctx.save_for_backward(k, u, a, b, axes, free)
        ctx.tol = tol  # type: ignore[attr-defined]
        ctx.max_iters = max_iters  # type: ignore[attr-defined]
        return u

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_u: Tensor
    ) -> tuple[Tensor | None, ...]:
        k, u, a, b, axes, free = ctx.saved_tensors  # type: ignore[attr-defined]
        with torch.no_grad():
            zero = torch.zeros((), dtype=grad_u.dtype, device=grad_u.device)
            rhs = torch.where(free, grad_u, zero)
            lam = _cg_solve(
                k,
                rhs,
                a,
                b,
                axes,
                free,
                ctx.tol,  # type: ignore[attr-defined]
                ctx.max_iters,  # type: ignore[attr-defined]
            )
            u3 = u.reshape(-1, 3)
            l3 = lam.reshape(-1, 3)
            s_u = (axes * (u3[a] - u3[b])).sum(dim=1)  # (E,)
            s_l = (axes * (l3[a] - l3[b])).sum(dim=1)  # (E,)
            grad_k = -(s_l * s_u)
            grad_f = torch.where(free, lam, zero)
        return grad_k, grad_f, None, None, None, None, None, None


def _solve_sparse(sheet: Sheet, k: Tensor, f: Tensor, free: Tensor, cfg: SimConfig) -> Tensor:
    """Matrix-free static solve; returns the full DOF displacement, shape ``(3N,)``."""
    dtype = sheet.rest.dtype
    tol = cfg.cg_tol if cfg.cg_tol is not None else (1e-12 if dtype == torch.float64 else 1e-6)
    max_iters = cfg.cg_max_iters if cfg.cg_max_iters is not None else int(free.sum())
    axes = _edge_axes(sheet)
    a, b = sheet.edges[:, 0], sheet.edges[:, 1]
    return _SparseStaticSolve.apply(k, f, a, b, axes, free, tol, max_iters)


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

    # External load: a total load split over the loaded nodes.
    f = torch.zeros((sheet.num_nodes, 3), dtype=dtype, device=device)
    n_loaded = int(sheet.loaded.sum())
    if n_loaded > 0:
        load_vec = torch.tensor(cfg.load, dtype=dtype, device=device) / n_loaded
        f = torch.where(sheet.loaded.unsqueeze(1), load_vec, f)
    f = f.reshape(-1)

    free = _free_dofs(sheet)
    # The solve must run in the sheet's precision even under autocast (AMP): an
    # ill-conditioned stiffness matrix solved in low precision destroys the
    # physical gradient. Disable autocast here and cast the operands back to `dtype`.
    amp_device = device.type if device.type in ("cpu", "cuda") else "cpu"
    with torch.autocast(device_type=amp_device, enabled=False):
        if cfg.sparse:
            # Matrix-free path: K is never assembled, memory stays O(N).
            u = _solve_sparse(sheet, k.to(dtype), f.to(dtype), free, cfg)
            compliance = f @ u
        else:
            k_global = _assemble_stiffness(sheet, k)
            k_ff = k_global[free][:, free]
            f_f = f[free]
            u_f = torch.linalg.solve(k_ff.to(dtype), f_f.to(dtype))
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
