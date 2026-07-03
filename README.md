<div align="center">

# Differentiable Digital Twin

**Gradient-based inverse design of a load-bearing structure — solved by backpropagating
through a physics solver, then replayed in the browser on WebGPU.**

[![Live demo](https://img.shields.io/badge/live%20demo-WebGPU%20player-D97706)](https://erik2810.github.io/differentiable-digital-twin/)
[![CI](https://img.shields.io/badge/CI-ruff%20%2B%20pytest-1E40AF)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-3B82F6)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-D97706)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-1E40AF)](LICENSE)

**▶ Live WebGPU demo: https://erik2810.github.io/differentiable-digital-twin/**

</div>

```
        L O A D                                                              ┌── PyTorch ──┐
          │                                                                  │  K(ρ)u = f  │  forward  solve
          ▼            optimised material layout  ρ*  (40% budget)           │   C = f·u   │
   ╓──────────────────────────────────────────────────────────╖             └──────┬──────┘
   ║ @@@@@@@@@@#.                                               ║                    │ ∂C/∂ρ   (one backward pass,
   ║ :@@@##@@@@@@@                                              ║                    ▼          cost independent of N)
   ║   +@@    #+@@@@#                                       @   ║             ┌─────────────┐
   ║     @@        #@@@@   @                                    ║   Adam ◀────│  autodiff   │
   ║      @@: @       @@@ @*            stiff material  ███     ║             └──────┬──────┘
   ║        @@         .@@@             soft / void     ░░░     ║                    │ ρ*, trajectory
   ║      @@: @       @@@ @*                                    ║                    ▼
   ║     @@        #@@@@   @                                    ║             ┌─────────────┐
   ║   +@@    #+@@@@#                                       @   ║             │  WebGPU     │  scene.json
   ║ :@@@##@@@@@@@                                              ║             │  player     │  positions.bin
   ║ @@@@@@@@@@#.                                               ║             └─────────────┘
   ╙──────────────────────────────────────────────────────────╜
    ▲ clamped edge                              free tip ▲
```

A compact, end-to-end demonstration that **a simulator you can differentiate turns engineering
design from guess-and-check into a gradient descent.** A deformable panel is clamped at one edge
and loaded at the other; the optimiser decides where to place a fixed budget of stiff material so
the structure is as rigid as possible. The result — a load-bearing truss discovered from a
uniform slab — is found in **~200 solves**, where a derivative-free search of the same design
space needs **thousands to tens of thousands**, and the gap widens without bound as the design
grows.

---

## TL;DR

| | |
|---|---|
| **Problem** | Distribute a 40% material budget over a clamped panel to minimise its deflection under a prescribed tip load (the canonical *compliance-minimisation* / topology-optimisation problem). |
| **Method** | Assemble the linear-elastic stiffness matrix `K(ρ)`, solve `K(ρ)u = f`, and minimise compliance `C = f·u` by **reverse-mode autodiff through the solve** (PyTorch). |
| **Result** | **4.6× stiffer** than a uniform design at the same material budget (compliance `1.461 → 0.317`, −78%), found in **~2 s** on a laptop CPU. |
| **Why it matters** | The gradient over *all* `N` design variables costs one extra solve. Black-box optimisers cost `O(N)` solves *per step* — the basis of the orders-of-magnitude speedup quantified [below](#the-industrial-thesis-gradients-vs-guessing). |
| **Frontend** | A dependency-free **WebGPU** player (Three.js, WebGL2 fallback) animates the optimised trajectory and the material field. |

---

## The industrial thesis: gradients vs. guessing

Classical engineering optimisation treats the simulator as a **black box**: pick a design,
simulate, score it, adjust, repeat. Every method built on that contract — grid search, genetic
algorithms, CMA-ES, Bayesian optimisation, finite differences — pays the same structural tax:
**to learn how `N` parameters affect the objective, you must perturb and re-simulate roughly `N`
times.** Design spaces in real structural problems have `N` in the tens of thousands to millions
of elements. That is why a single optimisation campaign can mean weeks of cluster time.

A **differentiable** simulator changes the contract. Reverse-mode automatic differentiation (the
*adjoint method*, rediscovered) computes the sensitivity of the objective to **every** design
variable at once, in a single backward pass whose cost is *independent of `N`*:

$$
\nabla_\rho C \;=\; \underbrace{\text{one backward pass}}_{\mathcal{O}(1)\ \text{solves}}
\qquad\text{vs.}\qquad
\nabla_\rho C \approx \Big[\tfrac{C(\rho+\epsilon e_i)-C(\rho)}{\epsilon}\Big]_{i=1}^{N}
\;=\; \underbrace{N{+}1\ \text{re-simulations}}_{\mathcal{O}(N)\ \text{solves}}
$$

For compliance there is even a closed form the adjoint method derives by hand,
$\partial C/\partial \rho_e = -\,u_e^\top (\partial K_e/\partial \rho_e)\, u_e$; here we simply
let autodiff compute it and never write it down.

### Measured head-to-head

Identical problem, identical budget penalty, all methods seeing the solver only as an oracle
`ρ → C(ρ)`. The accounting unit is one static solve. *(Reproduce with `ddtwin benchmark`.)*

**N = 60 design variables** (6×10 grid)

| method | final loss | static solves | wall time | vs. autodiff |
|---|--:|--:|--:|--:|
| **autodiff (Adam)** | **0.354** | **120** | **0.95 s** | **1.0×** |
| finite-diff GD | 0.387 | 2 441 | 1.20 s | 1.3× |
| CMA-ES | 0.285 | 2 448 | 1.31 s | 1.4× |
| random search | 0.620 | 2 441 | 1.15 s | 1.2× |

**N = 150 design variables** (10×15 grid)

| method | final loss | static solves | wall time | vs. autodiff |
|---|--:|--:|--:|--:|
| **autodiff (Adam)** | **0.237** | **120** | **0.91 s** | **1.0×** |
| finite-diff GD | 0.273 | 6 041 | 5.83 s | 6.4× |
| CMA-ES | 0.195 | 6 042 | 10.09 s | 11.1× |
| random search | 0.533 | 6 041 | 6.20 s | 6.8× |

**Read the scaling, not just the totals.** Autodiff reaches a comparable design at a *fixed* 120
solves regardless of `N`. The black-box methods' solve count tracks `N` exactly
(`steps × (N+1)`), so as `N` grows from 60 → 150 the wall-clock gap widens from ~1.3× to
6–11×. Extrapolate the `O(N)` line to an industrial mesh:

```
   solves to evaluate one gradient
   │
   │                                       finite-diff  ∝ (N+1)         ·
   │                                                              ·
   │                                                       ·
   │                                                ·
   │                                         ·
   │                                  ·
   │                           ·
   │                    ·
   │             ·
   │      ·
   │·________________________________________________________  autodiff  ≈ const
   └──────────────────────────────────────────────────────────▶  N (design variables)
     60     150        10³            10⁴            10⁵        10⁶
   1.3–1.4×  6–11×    ~10²×          ~10³×          ~10⁴×      ~10⁵×   ← projected wall-clock gap
```

That is the "orders of magnitude" claim, stated precisely: it is **asymptotic in the size of the
design space**, and it is exactly why differentiable simulation is reshaping structural
optimisation, inverse design, and digital-twin calibration. (CMA-ES finds a marginally lower loss
here because it is allowed 20× the solve budget — the point is that autodiff matches it using a
*constant, tiny fraction* of the simulations, and that fraction shrinks as the problem grows.)

---

## The physics & the inverse problem

The twin is a rectangular panel discretised on a `rows × cols` grid of nodes, connected by
linear-elastic two-force members in three families (structural / shear / bending). It is modelled
as a 2D in-plane elastic truss — the standard reduction for compliance optimisation.

**Design variable.** A per-node *material density* $\rho_i \in (0,1)$, parameterised by
unconstrained logits through a sigmoid so the optimiser runs unconstrained. Densities set member
stiffness through the **SIMP** interpolation (Solid Isotropic Material with Penalisation; Bendsøe
& Sigmund), whose exponent $p>1$ makes intermediate "grey" material inefficient and pushes the
optimum toward a crisp, manufacturable black/white layout:

$$
k_e(\rho) \;=\; k_{\min} + (k_{\max}-k_{\min})\,\bar\rho_e^{\,p},
\qquad \bar\rho_e = \tfrac12(\rho_a + \rho_b),\qquad p = 3.
$$

**Forward model.** Assemble the global stiffness matrix from member contributions
$k_e\,\hat d_e \hat d_e^\top$ and solve the linear system for the nodal displacement:

$$
K(\rho)\,u = f .
$$

**Objective.** Minimise the **compliance** (work done by the load = stored strain energy;
minimising it maximises stiffness), subject to a material budget:

$$
\min_{\rho}\; C(\rho) = f^\top u(\rho)
\quad\text{s.t.}\quad \tfrac1N\textstyle\sum_i \rho_i \le V,\;\; \rho_i \in (0,1),
$$

handled as an unconstrained loss with a quadratic budget penalty
$L(\theta) = C(\rho) + w\,\mathrm{relu}(\bar\rho - V)^2$, and optimised with Adam. Every iteration
is **one solve forward, one backward pass** — and `torch.linalg.solve` is differentiable, so the
full design gradient falls out automatically. A `gradcheck` against finite differences guards it
in CI.

---

## Three things this project demonstrates

1. **Differentiable physics.** A from-scratch linear-elastic FEM assembly and solve in PyTorch,
   differentiated end-to-end. No autograd-breaking in-place ops; verified by `torch.autograd.gradcheck`.
2. **WebGPU frontend engineering.** A dependency-free browser player (Three.js `WebGPURenderer`,
   automatic WebGL2 fallback) that streams a compact binary trajectory and renders the material
   field, the deflection (displacement-scaled, as an FEA post-processor would), and the load path.
3. **Production software hygiene.** `uv`-locked environment, `ruff` + `mypy` + `pytest`, a typed
   public API, a CPU-only multi-stage Docker image, and CI — the parts that separate a research
   script from a shippable framework.

---

## Quickstart

### Local (uv)

```bash
uv sync                       # create the locked environment (PyTorch + numpy)

uv run ddtwin optimise        # gradient-based inverse design -> exports to web/data/
uv run ddtwin benchmark       # reproduce the autodiff vs. black-box table
uv run pytest                 # run the suite (includes the gradcheck)
```

Then serve the player (a static server is required — `fetch` of the binary trajectory is blocked
on `file://`):

```bash
cd web && python -m http.server 8080      # open http://localhost:8080
```

Drag to orbit; scrub the timeline; raise **displacement scale** to exaggerate the deflection;
toggle the **wireframe** to see the truss members.

### Docker

The training framework is containerised as a slim, non-root, CPU-only image (the lockfile pins
the CPU PyTorch wheel on Linux):

```bash
docker build -t ddtwin .

# optimise inside the container; the trajectory lands in ./out for the player
docker run --rm -v "$PWD/out:/app/web/data" ddtwin optimise --float32
```

---

## CLI

```
ddtwin optimise  [--rows R --cols C] [--iters N] [--vol-frac V] [--float32] [--out DIR]
ddtwin benchmark [--rows R --cols C] [--iters N] [--fd-steps S] [--out FILE.json]
ddtwin simulate  [--rows R --cols C] [--vol-frac V] [--out DIR]
```

Running `ddtwin optimise` on the default 12×22 cantilever prints, e.g.:

```
[optimise] cantilever 12x22 (264 nodes, 1416 members), 200 iters
[optimise] done in 1.9s
           compliance     : 1.4607 -> 0.3167  (78.3% lower)
           tip deflection : 1.4835 -> 0.3213  (78.3% lower)
```

---

## Repository layout

```
differentiable-digital-twin/
├── src/ddtwin/
│   ├── geometry.py        # cantilever mesh + spring topology (Sheet)
│   ├── materials.py       # density field, SIMP parameterisation (MaterialField)
│   ├── physics.py         # differentiable static FEM: assemble K(ρ), solve K u = f
│   ├── optimize.py        # compliance minimisation with Adam + autodiff
│   ├── baselines.py       # black-box optimisers: finite-diff, random, CMA-ES
│   ├── export.py          # trajectory -> scene.json + positions.bin
│   └── cli.py             # `ddtwin` entry point (optimise / benchmark / simulate)
├── tests/                 # pytest, incl. gradcheck through the solve
├── web/                   # WebGPU player (index.html · player.js · style.css)
│   └── data/              # exported scene.json + positions.bin
├── Dockerfile             # multi-stage, CPU-only, non-root
├── pyproject.toml         # uv-managed; ruff + mypy + pytest config
├── ARCHITECTURE.md        # data flow and design notes
└── CITATION.cff
```

---

## Honest scope

This is a *focused demonstrator*, and its modelling choices are deliberate and stated:

- The structure is a **linear-elastic in-plane truss** (small-deflection, plane reduction). It is
  the right model for compliance optimisation and keeps the maths legible; it is not large-strain
  contact dynamics.
- Node displacements are physically small, so the player **exaggerates** them for visualisation
  (a labelled "displacement scale", exactly as ANSYS/Abaqus post-processors do by default).
- The browser animation is a synthesised load-application ramp toward the computed static
  solution `u`, not a re-simulation in the browser — the player *reads* the optimised tensors, as
  intended.

None of this weakens the thesis: the gradient-vs-black-box scaling argument is a property of the
*method*, not the model, and holds identically for a nonlinear or 3D solver.

## Roadmap

- [ ] Nonlinear (large-deflection) solver with a differentiable Newton loop
- [ ] Density filtering + projection for mesh-independent, manufacturable layouts
- [ ] Multiple load cases and stress constraints
- [ ] GPU-side trajectory playback via a WebGPU compute shader (TSL)

## References

- M. P. Bendsøe and O. Sigmund, *Topology Optimization: Theory, Methods, and Applications*, Springer, 2003.
- A. Griewank and A. Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, SIAM, 2008.
- Hu et al., *DiffTaichi: Differentiable Programming for Physical Simulation*, ICLR 2020.

## License

MIT — see [LICENSE](LICENSE). Part of a computational-physics portfolio; companion to the
JAX/PyTorch differentiable-physics and WebGPU mesh projects.
