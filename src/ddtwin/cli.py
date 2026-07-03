"""Command-line entry point: ``ddtwin <command>``.

Commands
--------
optimise    Run gradient-based inverse design and export a trajectory for the web player.
benchmark   Compare autodiff vs. black-box optimisers and write a results table.
simulate    Solve a uniform design (a quick smoke test / reference) and optionally export it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ddtwin.baselines import cma_es, finite_difference_descent, random_search
from ddtwin.geometry import Sheet
from ddtwin.optimize import InverseDesignConfig, inverse_design
from ddtwin.physics import SimConfig, simulate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WEB_DATA = _REPO_ROOT / "web" / "data"


@dataclass
class _BenchRow:
    method: str
    loss: float
    sims: int
    wall_s: float


def _build_sheet(args: argparse.Namespace, dtype: torch.dtype) -> Sheet:
    return Sheet.cantilever(rows=args.rows, cols=args.cols, device="cpu", dtype=dtype)


def _cmd_optimise(args: argparse.Namespace) -> None:
    from ddtwin.export import export_scene

    dtype = torch.float32 if args.float32 else torch.float64
    sheet = _build_sheet(args, dtype)
    cfg = InverseDesignConfig(
        iters=args.iters,
        lr=args.lr,
        vol_frac=args.vol_frac,
        sim=SimConfig(frames=args.frames),
    )
    print(
        f"[optimise] cantilever {args.rows}x{args.cols} ({sheet.num_nodes} nodes, "
        f"{sheet.num_edges} members), {cfg.iters} iters"
    )
    start = time.perf_counter()
    result = inverse_design(sheet, cfg, progress=True)
    wall = time.perf_counter() - start

    c0, c1 = result.uniform_compliance, result.compliance
    d0, d1 = result.uniform_tip_deflection, result.tip_deflection
    ci, di = 100 * result.improvement, 100 * result.deflection_improvement
    print(f"[optimise] done in {wall:.1f}s")
    print(f"           compliance     : {c0:.4f} -> {c1:.4f}  ({ci:.1f}% lower)")
    print(f"           tip deflection : {d0:.4f} -> {d1:.4f}  ({di:.1f}% lower)")

    out = Path(args.out) if args.out else _WEB_DATA
    export_scene(out, sheet, result, cfg)
    print(f"[optimise] trajectory exported to {out}")


def _cmd_simulate(args: argparse.Namespace) -> None:
    from ddtwin.export import export_scene
    from ddtwin.optimize import InverseDesignResult

    dtype = torch.float32 if args.float32 else torch.float64
    sheet = _build_sheet(args, dtype)
    sim = SimConfig(frames=args.frames)
    density = torch.full((sheet.num_nodes,), args.vol_frac, dtype=dtype)
    res = simulate(sheet, density, sim)
    compliance = float(res.compliance())
    deflection = float(res.tip_deflection())
    print(f"[simulate] uniform rho={args.vol_frac}: C={compliance:.4f} deflection={deflection:.4f}")
    if args.out:
        cfg = InverseDesignConfig(vol_frac=args.vol_frac, sim=sim)
        wrapped = InverseDesignResult(
            density=density,
            logits=torch.logit(density),
            loss_history=[compliance],
            compliance_history=[compliance],
            volume_history=[float(density.mean())],
            sim_calls=1,
            final=res,
            uniform_compliance=compliance,
            uniform_tip_deflection=deflection,
        )
        export_scene(args.out, sheet, wrapped, cfg)
        print(f"[simulate] trajectory exported to {args.out}")


def _cmd_benchmark(args: argparse.Namespace) -> None:
    # A small mesh keeps the O(N)-per-step black-box methods tractable.
    sheet = Sheet.cantilever(rows=args.rows, cols=args.cols, device="cpu", dtype=torch.float32)
    sim = SimConfig()
    n = sheet.num_nodes
    print(f"[benchmark] cantilever {args.rows}x{args.cols} -> {n} design variables\n")

    # --- Gradient-based (autodiff through the static solve).
    cfg = InverseDesignConfig(iters=args.iters, sim=sim)
    t0 = time.perf_counter()
    grad = inverse_design(sheet, cfg)
    grad_wall = time.perf_counter() - t0
    rows = [_BenchRow("autodiff (Adam)", grad.loss_history[-1], grad.sim_calls, grad_wall)]

    # --- Finite-difference descent.
    fd = finite_difference_descent(sheet, sim, steps=args.fd_steps)
    rows.append(_BenchRow("finite-diff GD", fd.best_loss, fd.sim_calls, fd.wall_time))

    # --- Random search (matched to the finite-diff solve budget).
    rs = random_search(sheet, sim, samples=fd.sim_calls)
    rows.append(_BenchRow("random search", rs.best_loss, rs.sim_calls, rs.wall_time))

    # --- CMA-ES (optional).
    cma = cma_es(sheet, sim, max_evals=fd.sim_calls)
    if cma is not None:
        rows.append(_BenchRow("CMA-ES", cma.best_loss, cma.sim_calls, cma.wall_time))

    # --- Report.
    header = (
        f"{'method':<20}{'final loss':>14}{'static solves':>16}{'wall (s)':>12}{'slowdown x':>12}"
    )
    print(header)
    print("-" * len(header))
    grad_wall_ref = rows[0].wall_s or 1e-9
    for r in rows:
        print(
            f"{r.method:<20}{r.loss:>14.4f}{r.sims:>16d}"
            f"{r.wall_s:>12.2f}{r.wall_s / grad_wall_ref:>12.1f}"
        )

    if args.out:
        payload = {"nodes": n, "rows": [asdict(r) for r in rows]}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\n[benchmark] results written to {args.out}")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rows", type=int, default=12)
    p.add_argument("--cols", type=int, default=22)
    p.add_argument("--frames", type=int, default=120, help="animation frames in the export")
    p.add_argument("--vol-frac", type=float, default=0.4, dest="vol_frac")
    p.add_argument("--float32", action="store_true", help="use float32 (faster; default float64)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ddtwin", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_opt = sub.add_parser("optimise", help="gradient-based inverse design + export")
    _common(p_opt)
    p_opt.add_argument("--iters", type=int, default=160)
    p_opt.add_argument("--lr", type=float, default=0.08)
    p_opt.add_argument("--out", type=str, default=None, help="export dir (default web/data)")
    p_opt.set_defaults(func=_cmd_optimise)

    p_sim = sub.add_parser("simulate", help="solve a uniform design")
    _common(p_sim)
    p_sim.add_argument("--out", type=str, default=None)
    p_sim.set_defaults(func=_cmd_simulate)

    p_bench = sub.add_parser("benchmark", help="autodiff vs black-box comparison")
    p_bench.add_argument("--rows", type=int, default=6)
    p_bench.add_argument("--cols", type=int, default=10)
    p_bench.add_argument("--iters", type=int, default=120)
    p_bench.add_argument("--fd-steps", type=int, default=40, dest="fd_steps")
    p_bench.add_argument("--out", type=str, default=None)
    p_bench.set_defaults(func=_cmd_benchmark)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
