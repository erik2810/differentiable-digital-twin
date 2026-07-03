"""Gradient-based inverse design (compliance minimisation).

We solve the constrained topology-optimisation problem

    minimise    C(rho) = f . u(rho)     (compliance: work done by the load = strain energy)
    subject to  mean(rho) <= V          (material budget / volume fraction)
                rho in (0, 1)

as an unconstrained problem in logits ``theta`` (with ``rho = sigmoid(theta)``) plus a quadratic
penalty on the budget:

    L(theta) = C(rho) + w * relu(mean(rho) - V)**2

Each iteration is one static solve (forward) and one backward pass. The gradient ``dL/dtheta``
spans all ``N`` design variables yet costs the same as a single extra solve -- the property the
black-box baselines in :mod:`ddtwin.baselines` fundamentally cannot match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ddtwin.geometry import Sheet
from ddtwin.materials import MaterialField
from ddtwin.physics import SimConfig, SimResult, simulate, solve_static


@dataclass
class InverseDesignConfig:
    iters: int = 160
    lr: float = 0.08
    vol_frac: float = 0.4  # material budget V
    budget_weight: float = 60.0  # penalty weight w
    init_density: float = 0.4
    seed: int = 0
    log_every: int = 20
    sim: SimConfig = field(default_factory=SimConfig)


@dataclass
class InverseDesignResult:
    density: Tensor  # (N,) optimised density
    logits: Tensor  # (N,) optimised logits
    loss_history: list[float]
    compliance_history: list[float]
    volume_history: list[float]
    sim_calls: int  # static solves consumed (== iters for the gradient method)
    final: SimResult
    uniform_compliance: float  # compliance of a uniform design at the same budget (reference)
    uniform_tip_deflection: float

    @property
    def compliance(self) -> float:
        return self.compliance_history[-1]

    @property
    def tip_deflection(self) -> float:
        return float(self.final.tip_deflection())

    @property
    def improvement(self) -> float:
        """Fractional reduction in compliance vs. the uniform reference design."""
        if self.uniform_compliance == 0.0:
            return 0.0
        return 1.0 - self.compliance / self.uniform_compliance

    @property
    def deflection_improvement(self) -> float:
        if self.uniform_tip_deflection == 0.0:
            return 0.0
        return 1.0 - self.tip_deflection / self.uniform_tip_deflection


def _uniform_reference(sheet: Sheet, cfg: InverseDesignConfig) -> tuple[float, float]:
    """Compliance and tip deflection of a uniform design that spends the budget evenly."""
    with torch.no_grad():
        density = torch.full(
            (sheet.num_nodes,), cfg.vol_frac, dtype=sheet.rest.dtype, device=sheet.device
        )
        res = solve_static(sheet, density, cfg.sim)
        return float(res.compliance), float(res.tip_deflection())


def inverse_design(
    sheet: Sheet, cfg: InverseDesignConfig | None = None, *, progress: bool = False
) -> InverseDesignResult:
    """Optimise the material distribution with Adam + autodiff through the static solve."""
    cfg = cfg or InverseDesignConfig()
    torch.manual_seed(cfg.seed)

    field_ = MaterialField(sheet, init=cfg.init_density)
    optimiser = torch.optim.Adam(field_.parameters, lr=cfg.lr)

    loss_history: list[float] = []
    compliance_history: list[float] = []
    volume_history: list[float] = []

    for step in range(cfg.iters):
        optimiser.zero_grad()
        density = field_.density()
        result = solve_static(sheet, density, cfg.sim)
        compliance = result.compliance
        over_budget = torch.relu(density.mean() - cfg.vol_frac)
        loss = compliance + cfg.budget_weight * over_budget**2
        loss.backward()
        optimiser.step()

        loss_history.append(loss.item())
        compliance_history.append(compliance.item())
        volume_history.append(density.mean().item())
        if progress and (step % cfg.log_every == 0 or step == cfg.iters - 1):
            print(
                f"  iter {step:4d}  loss {loss.item():.4f}  "
                f"compliance {compliance.item():.4f}  vol {density.mean().item():.3f}"
            )

    with torch.no_grad():
        final_density = field_.density().detach()
    final = simulate(sheet, final_density, cfg.sim)
    uniform_c, uniform_d = _uniform_reference(sheet, cfg)

    return InverseDesignResult(
        density=final_density,
        logits=field_.logits.detach(),
        loss_history=loss_history,
        compliance_history=compliance_history,
        volume_history=volume_history,
        sim_calls=cfg.iters,
        final=final,
        uniform_compliance=uniform_c,
        uniform_tip_deflection=uniform_d,
    )
