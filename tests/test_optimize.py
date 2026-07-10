"""Inverse-design loop: the optimiser must beat a uniform design and respect the budget."""

from __future__ import annotations

import torch

from ddtwin.geometry import Sheet
from ddtwin.optimize import InverseDesignConfig, inverse_design


def test_optimisation_reduces_compliance() -> None:
    sheet = Sheet.cantilever(rows=6, cols=12, dtype=torch.float32)
    cfg = InverseDesignConfig(iters=120, vol_frac=0.4)
    res = inverse_design(sheet, cfg)
    # Loss drops well below the start.
    assert res.loss_history[-1] < res.loss_history[0]
    # The optimised design is markedly stiffer than the uniform reference at the same budget.
    assert res.compliance < res.uniform_compliance
    assert res.improvement > 0.3
    # Tip deflection improves too.
    assert res.tip_deflection < res.uniform_tip_deflection


def test_budget_is_respected() -> None:
    sheet = Sheet.cantilever(rows=6, cols=12, dtype=torch.float32)
    cfg = InverseDesignConfig(iters=140, vol_frac=0.35)
    res = inverse_design(sheet, cfg)
    assert res.volume_history[-1] <= 0.35 + 0.03


def test_density_in_unit_interval() -> None:
    sheet = Sheet.cantilever(rows=5, cols=10, dtype=torch.float32)
    res = inverse_design(sheet, InverseDesignConfig(iters=40))
    assert float(res.density.min()) >= 0.0
    assert float(res.density.max()) <= 1.0


def test_checkpoint_reproduces_baseline() -> None:
    # Gradient checkpointing recomputes the solve in backward; it is a pure
    # memory/compute trade and must not change the optimisation at all.
    sheet = Sheet.cantilever(rows=6, cols=10, dtype=torch.float64)
    base = inverse_design(sheet, InverseDesignConfig(iters=25, seed=0))
    ckpt = inverse_design(sheet, InverseDesignConfig(iters=25, seed=0, use_checkpoint=True))
    assert ckpt.loss_history == base.loss_history


def test_amp_is_stable() -> None:
    # AMP keeps the linear solve in the sheet's precision, so the run stays finite
    # and still reduces the loss (on float64 autocast is a no-op, so it matches too).
    import math

    sheet = Sheet.cantilever(rows=6, cols=10, dtype=torch.float64)
    res = inverse_design(
        sheet, InverseDesignConfig(iters=25, seed=0, use_checkpoint=True, use_amp=True)
    )
    assert all(math.isfinite(v) for v in res.loss_history)
    assert res.loss_history[-1] < res.loss_history[0]
