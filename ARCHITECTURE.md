# Architecture

The system has two halves connected by a small on-disk contract: a **Python/PyTorch backend**
that does the numerics, and a **JavaScript/WebGPU frontend** that does the rendering. This mirrors
the "Python compute, JS view" split used across the portfolio.

```
   ┌──────────────────────────── backend (PyTorch, CPU/GPU) ────────────────────────────┐
   │                                                                                     │
   │  geometry.Sheet ──▶ materials.MaterialField ──▶ physics.solve_static ──▶ compliance │
   │   mesh + springs        ρ = σ(θ), SIMP            assemble K(ρ);            C = f·u  │
   │                                                   solve K u = f                     │
   │                                                        │                            │
   │                                                        ▼  (autodiff)                │
   │                              optimize.inverse_design  ∂C/∂θ  ──▶ Adam ──▶ ρ*        │
   │                                                        │                            │
   │                                   physics.simulate(ρ*) │  load-ramp trajectory      │
   │                                                        ▼                            │
   │                                            export.export_scene                      │
   └────────────────────────────────────────────┬────────────────────────────────────────┘
                                                 │  scene.json  (topology, density, metadata)
                                                 │  positions.bin  (float32 frames×N×3)
                                                 ▼
   ┌──────────────────────────── frontend (web/, no build step) ────────────────────────┐
   │  player.js: fetch ─▶ THREE.WebGPURenderer (WebGL2 fallback) ─▶ animate mesh + HUD   │
   └─────────────────────────────────────────────────────────────────────────────────────┘
```

## Backend module map

| Module | Responsibility | Key types |
|---|---|---|
| `geometry.py` | Build the cantilever grid: node positions, spring members (structural/shear/bend), boundary conditions, render faces. | `Sheet` |
| `materials.py` | The learnable density field `ρ = σ(θ)` and SIMP interpolation parameters. | `MaterialField` |
| `physics.py` | Assemble `K(ρ)`, solve `K u = f` (`solve_static`), and synthesise a settling animation (`simulate`). | `SimConfig`, `StaticResult`, `SimResult` |
| `optimize.py` | Compliance minimisation under a budget penalty via Adam + autodiff. | `InverseDesignConfig`, `InverseDesignResult` |
| `baselines.py` | Derivative-free optimisers sharing one objective oracle, for benchmarking. | `BaselineResult` |
| `export.py` | Serialise the optimised trajectory to `scene.json` + `positions.bin`. | — |
| `cli.py` | `ddtwin` entry point. | — |

## Why a static linear solve (not a time-stepped sim)

Compliance minimisation is a *static equilibrium* problem: the design objective is a functional
of `u = K(ρ)⁻¹ f`. Solving it directly (one linear solve) is exact, fast, and — crucially —
yields the textbook truss layouts, because the optimum is `argmin_ρ f^⊤ K(ρ)⁻¹ f` under the
volume constraint. Differentiating through `torch.linalg.solve` gives the design gradient for
free. A long dynamic relaxation would only approximate this equilibrium (mass-spring sheets have a
very soft fundamental mode that never fully settles) and would let the optimiser exploit
transient dynamics rather than build a genuine load path.

The visible animation in the player is a *separate* concern: a synthesised load-application ramp
toward the computed static `u`, generated once after optimisation purely for display.

## The on-disk contract

- **`scene.json`** — `meta` (grid shape, node/frame counts), `topology` (render faces, spring
  edges, rest positions), `boundary` (clamped/loaded masks, load vector), `design` (the optimised
  density field, compliance, deflection, improvement), and the optimisation `history`.
- **`positions.bin`** — raw little-endian `float32`, shape `(frames, N, 3)`, row-major. Binary
  keeps it ~6× smaller than JSON and parses straight into a typed array — the same
  binary-protocol convention used for streaming elsewhere in the portfolio, here at rest.

## Differentiability discipline

The single most common bug in differentiable physics is an in-place op silently severing the
autograd graph. The solver avoids it: forces/stiffness are accumulated with out-of-place
`index_add`, boundary conditions are applied by masked indexing into freshly allocated tensors,
and `tests/test_physics.py` runs `torch.autograd.gradcheck` on the compliance through the full
solve in float64 on every CI run.
