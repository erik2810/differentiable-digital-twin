"""Black-box (derivative-free) baselines for the inverse-design problem.

These exist to quantify what gradient-based design *buys* you. Each baseline sees the solver
only as an oracle ``theta -> L(theta)`` and may never look inside it. Three strategies are
provided:

* ``random_search``            -- i.i.d. samples, keep the best.
* ``finite_difference_descent`` -- estimate the gradient with forward differences
  (``N + 1`` forward sims per step) then take a descent step.
* ``cma_es``                   -- CMA-ES via the optional ``cma`` package.

The accounting unit is the **static solve**: every call to :func:`evaluate` runs one
``K u = f`` solve. The gradient method in :mod:`ddtwin.optimize` reaches a good design in ~`iters`
backward passes, whereas finite differences need ``O(N)`` solves *per step* -- the gap grows
linearly with the number of design variables ``N``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ddtwin.geometry import Sheet
from ddtwin.physics import SimConfig, solve_static


@dataclass
class BaselineResult:
    name: str
    density: Tensor
    best_loss: float
    loss_history: list[float]  # best-so-far loss per logged step
    sim_calls: int
    wall_time: float


def make_objective(
    sheet: Sheet, sim: SimConfig, vol_frac: float, budget_weight: float
) -> tuple[Callable[[Tensor], float], Callable[[], int]]:
    """Return ``(evaluate, counter)``.

    ``evaluate(theta)`` maps logits to the scalar penalised loss (one forward sim) and
    increments an internal counter that ``counter()`` reads back.
    """
    calls = {"n": 0}

    def evaluate(theta: Tensor) -> float:
        calls["n"] += 1
        with torch.no_grad():
            density = torch.sigmoid(theta.to(dtype=sheet.rest.dtype, device=sheet.device))
            result = solve_static(sheet, density, sim)
            compliance = result.compliance
            over = torch.relu(density.mean() - vol_frac)
            return float(compliance + budget_weight * over**2)

    def counter() -> int:
        return calls["n"]

    return evaluate, counter


def random_search(
    sheet: Sheet,
    sim: SimConfig,
    *,
    vol_frac: float = 0.4,
    budget_weight: float = 60.0,
    samples: int = 600,
    scale: float = 1.5,
    seed: int = 0,
) -> BaselineResult:
    """Sample logits ~ N(logit(V), scale) and keep the best."""
    torch.manual_seed(seed)
    evaluate, counter = make_objective(sheet, sim, vol_frac, budget_weight)
    mean = torch.logit(torch.tensor(vol_frac))
    best_theta = torch.full((sheet.num_nodes,), float(mean))
    best_loss = float("inf")
    history: list[float] = []

    start = time.perf_counter()
    for _ in range(samples):
        theta = mean + scale * torch.randn(sheet.num_nodes)
        loss = evaluate(theta)
        if loss < best_loss:
            best_loss, best_theta = loss, theta
        history.append(best_loss)
    wall = time.perf_counter() - start

    return BaselineResult(
        name="random_search",
        density=torch.sigmoid(best_theta),
        best_loss=best_loss,
        loss_history=history,
        sim_calls=counter(),
        wall_time=wall,
    )


def finite_difference_descent(
    sheet: Sheet,
    sim: SimConfig,
    *,
    vol_frac: float = 0.4,
    budget_weight: float = 60.0,
    steps: int = 40,
    lr: float = 0.6,
    eps: float = 1e-2,
    init_density: float = 0.4,
    seed: int = 0,
) -> BaselineResult:
    """Forward-difference gradient descent.

    Each step probes every coordinate once (``N + 1`` forward sims) to build a gradient
    estimate, then takes an Adam-style step. This is the fairest black-box analogue of the
    autodiff method -- and exposes its ``O(N)``-per-step cost.
    """
    torch.manual_seed(seed)
    evaluate, counter = make_objective(sheet, sim, vol_frac, budget_weight)
    n = sheet.num_nodes
    theta = torch.full((n,), float(torch.logit(torch.tensor(init_density))))

    # Lightweight Adam state.
    m = torch.zeros(n)
    v = torch.zeros(n)
    beta1, beta2, edenom = 0.9, 0.999, 1e-8
    history: list[float] = []

    start = time.perf_counter()
    for t in range(1, steps + 1):
        base = evaluate(theta)
        grad = torch.empty(n)
        for i in range(n):
            probe = theta.clone()
            probe[i] += eps
            grad[i] = (evaluate(probe) - base) / eps
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta = theta - lr * m_hat / (v_hat.sqrt() + edenom)
        history.append(base)
    wall = time.perf_counter() - start

    best_loss = evaluate(theta)
    return BaselineResult(
        name="finite_difference",
        density=torch.sigmoid(theta),
        best_loss=best_loss,
        loss_history=history,
        sim_calls=counter(),
        wall_time=wall,
    )


def cma_es(
    sheet: Sheet,
    sim: SimConfig,
    *,
    vol_frac: float = 0.4,
    budget_weight: float = 60.0,
    max_evals: int = 3000,
    sigma0: float = 1.2,
    seed: int = 0,
) -> BaselineResult | None:
    """CMA-ES baseline. Returns ``None`` if the optional ``cma`` package is missing."""
    try:
        import cma
    except ImportError:
        return None

    evaluate, counter = make_objective(sheet, sim, vol_frac, budget_weight)
    x0 = [float(torch.logit(torch.tensor(vol_frac)))] * sheet.num_nodes
    es = cma.CMAEvolutionStrategy(x0, sigma0, {"seed": seed, "maxfevals": max_evals, "verbose": -9})
    history: list[float] = []
    best_loss = float("inf")

    start = time.perf_counter()
    while not es.stop():
        solutions = es.ask()
        losses = [evaluate(torch.tensor(s, dtype=torch.float32)) for s in solutions]
        es.tell(solutions, losses)
        best_loss = min(best_loss, min(losses))
        history.append(best_loss)
    wall = time.perf_counter() - start

    best = torch.tensor(es.result.xbest, dtype=torch.float32)
    return BaselineResult(
        name="cma_es",
        density=torch.sigmoid(best),
        best_loss=es.result.fbest,
        loss_history=history,
        sim_calls=counter(),
        wall_time=wall,
    )
