# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-06-29

### Added

- Differentiable static structural solver (`ddtwin.physics`): linear-elastic truss assembly of
  `K(ρ)` and `torch.linalg.solve` of `K u = f`, fully differentiable w.r.t. the material density.
- Gradient-based inverse design (`ddtwin.optimize`): compliance minimisation under a material
  budget via Adam + reverse-mode autodiff through the solve.
- Black-box baselines (`ddtwin.baselines`): finite-difference descent, random search, and optional
  CMA-ES, sharing one objective oracle for fair head-to-head benchmarking.
- SIMP material parameterisation (`ddtwin.materials`) and the cantilever mesh (`ddtwin.geometry`).
- Trajectory export (`ddtwin.export`): compact `scene.json` + binary `positions.bin`.
- WebGPU browser player (`web/`): Three.js `WebGPURenderer` with WebGL2 fallback, density colour
  map, displacement scaling, load/support overlays, and playback transport.
- `ddtwin` CLI: `optimise`, `benchmark`, `simulate`.
- Multi-stage, CPU-only, non-root `Dockerfile`; `uv` lockfile pins the CPU PyTorch wheel on Linux.
- Test suite with `torch.autograd.gradcheck` through the solver; `ruff` + `mypy` config; CI.

[0.1.0]: https://github.com/erik2810/differentiable-digital-twin/releases/tag/v0.1.0
