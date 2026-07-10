"""Static-solver behaviour and -- most importantly -- gradient correctness.

Autograd silently breaking is the classic differentiable-physics bug, so
:func:`torch.autograd.gradcheck` of the compliance through the linear solve is the
load-bearing test.
"""

from __future__ import annotations

import torch

from ddtwin.geometry import Sheet
from ddtwin.physics import SimConfig, simulate, solve_static


def _small_sheet() -> Sheet:
    return Sheet.cantilever(rows=4, cols=6, dtype=torch.float64)


def test_solve_is_finite_and_deforms() -> None:
    sheet = _small_sheet()
    density = torch.full((sheet.num_nodes,), 0.4, dtype=torch.float64)
    res = solve_static(sheet, density, SimConfig())
    assert torch.isfinite(res.u).all()
    assert float(res.compliance) > 0.0
    assert float(res.tip_deflection()) > 1e-4


def test_pinned_and_out_of_plane_dofs_are_fixed() -> None:
    sheet = _small_sheet()
    density = torch.full((sheet.num_nodes,), 0.5, dtype=torch.float64)
    res = solve_static(sheet, density, SimConfig())
    # Clamped nodes do not move.
    assert torch.allclose(
        res.u[sheet.pinned], torch.zeros(int(sheet.pinned.sum()), 3, dtype=torch.float64)
    )
    # The problem is in-plane: no out-of-plane (y) displacement anywhere.
    assert torch.allclose(res.u[:, 1], torch.zeros(sheet.num_nodes, dtype=torch.float64))


def test_stiffer_material_lowers_compliance() -> None:
    sheet = _small_sheet()
    cfg = SimConfig()
    soft = solve_static(sheet, torch.full((sheet.num_nodes,), 0.2, dtype=torch.float64), cfg)
    stiff = solve_static(sheet, torch.full((sheet.num_nodes,), 0.9, dtype=torch.float64), cfg)
    assert float(stiff.compliance) < float(soft.compliance)


def test_animation_starts_at_rest_and_ends_deformed() -> None:
    sheet = _small_sheet()
    density = torch.full((sheet.num_nodes,), 0.4, dtype=torch.float64)
    res = simulate(sheet, density, SimConfig(frames=60, overshoot=0.0))
    assert torch.allclose(res.positions[0], sheet.rest)  # envelope(0) = 0
    assert float((res.positions[-1] - sheet.rest).norm()) > 1e-3


def test_gradcheck_compliance_through_solve() -> None:
    """Differentiate compliance w.r.t. the density field through the linear solve."""
    torch.manual_seed(0)
    sheet = _small_sheet()
    cfg = SimConfig()

    def objective(density: torch.Tensor) -> torch.Tensor:
        return solve_static(sheet, density, cfg).compliance

    density = torch.rand(sheet.num_nodes, dtype=torch.float64, requires_grad=True) * 0.6 + 0.2
    assert torch.autograd.gradcheck(objective, (density,), atol=1e-4, rtol=1e-3, eps=1e-6)


def _random_density(sheet: Sheet, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(sheet.num_nodes, dtype=torch.float64) * 0.6 + 0.2


def test_sparse_solve_matches_dense() -> None:
    # The matrix-free CG path must reproduce the dense factorisation.
    sheet = _small_sheet()
    density = _random_density(sheet)
    dense = solve_static(sheet, density, SimConfig())
    sparse = solve_static(sheet, density, SimConfig(sparse=True))
    assert torch.allclose(sparse.u, dense.u, atol=1e-10)
    assert torch.allclose(sparse.compliance, dense.compliance, atol=1e-10)


def test_sparse_gradient_matches_dense() -> None:
    # The hand-written adjoint must agree with autodiff through the dense solve.
    sheet = _small_sheet()
    base = _random_density(sheet)
    d_dense = base.clone().requires_grad_(True)
    d_sparse = base.clone().requires_grad_(True)
    solve_static(sheet, d_dense, SimConfig()).compliance.backward()
    solve_static(sheet, d_sparse, SimConfig(sparse=True)).compliance.backward()
    assert d_dense.grad is not None and d_sparse.grad is not None
    assert torch.allclose(d_sparse.grad, d_dense.grad, atol=1e-10)


def test_gradcheck_compliance_through_sparse_solve() -> None:
    """The load-bearing test for the sparse path: the adjoint vs finite differences."""
    torch.manual_seed(0)
    sheet = _small_sheet()
    cfg = SimConfig(sparse=True)

    def objective(density: torch.Tensor) -> torch.Tensor:
        return solve_static(sheet, density, cfg).compliance

    density = torch.rand(sheet.num_nodes, dtype=torch.float64, requires_grad=True) * 0.6 + 0.2
    assert torch.autograd.gradcheck(objective, (density,), atol=1e-4, rtol=1e-3, eps=1e-6)


def test_sparse_pinned_and_out_of_plane_dofs_are_fixed() -> None:
    sheet = _small_sheet()
    density = torch.full((sheet.num_nodes,), 0.5, dtype=torch.float64)
    res = solve_static(sheet, density, SimConfig(sparse=True))
    assert torch.allclose(
        res.u[sheet.pinned], torch.zeros(int(sheet.pinned.sum()), 3, dtype=torch.float64)
    )
    assert torch.allclose(res.u[:, 1], torch.zeros(sheet.num_nodes, dtype=torch.float64))
