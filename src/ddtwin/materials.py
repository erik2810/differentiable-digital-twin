"""Material parameterisation.

The design variable is a per-node *material density* ``rho in (0, 1)`` -- the fraction
of stiff material allocated to each node. We parameterise it with unconstrained logits
``theta`` through a sigmoid so the optimiser can run unconstrained:

    rho = sigmoid(theta)

Densities map to spring stiffness through a SIMP-style interpolation (Solid Isotropic
Material with Penalisation, Bendsoe & Sigmund 2003):

    k(rho) = k_min + (k_max - k_min) * rho**p

The penalisation exponent ``p > 1`` makes intermediate densities inefficient, pushing the
optimum toward a crisp black/white (material / void) layout -- exactly what a manufacturing
process can realise.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ddtwin.geometry import Sheet


def logits_for_density(value: float) -> float:
    """Inverse sigmoid: the logit whose sigmoid equals ``value``."""
    value = min(max(value, 1e-4), 1.0 - 1e-4)
    return float(torch.logit(torch.tensor(value)))


class MaterialField:
    """Learnable per-node density field, stored as logits.

    Holds only the design variables; the stiffness/mass interpolation lives in
    :mod:`ddtwin.physics` so the forward solver stays a pure function of ``density``
    (which keeps :func:`torch.autograd.gradcheck` simple).
    """

    def __init__(self, sheet: Sheet, init: float = 0.4) -> None:
        self.sheet = sheet
        logit = logits_for_density(init)
        self.logits: Tensor = torch.full(
            (sheet.num_nodes,),
            logit,
            dtype=sheet.rest.dtype,
            device=sheet.device,
            requires_grad=True,
        )

    def density(self) -> Tensor:
        """Current density field ``rho in (0, 1)``, shape ``(N,)``."""
        return torch.sigmoid(self.logits)

    def volume_fraction(self) -> Tensor:
        """Mean of the ``(N,)`` density field -- the fraction of the budget currently used."""
        return self.density().mean()

    @property
    def parameters(self) -> list[Tensor]:
        return [self.logits]
