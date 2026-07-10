"""Profile the memory of the inverse-design backward pass, and verify stability.

The static solver assembles a *dense* global stiffness matrix of shape
``(3N, 3N)`` and factorises it, so its memory grows like ``O(N^2)`` and is what
makes a high-resolution mesh run out of memory. This script does three things,
matching the optimisation task:

1. Reports where the memory goes: the analytic size of the dense stiffness matrix
   as the mesh grows, next to the measured peak process RSS of one
   forward+backward pass.
2. Measures that peak with and without gradient checkpointing, so the
   compute-for-memory trade is visible. Each configuration runs in its own
   subprocess so ``ru_maxrss`` (a process-lifetime high-water mark) is clean.
3. Runs a short training (5 iterations) with checkpointing and AMP enabled and
   checks it stays finite and the loss decreases.

CUDA note: this machine has no CUDA GPU, so the run is on CPU. Gradient
checkpointing and the stability check are fully exercised here; the AMP path is
present and correct but only accelerates on a CUDA GPU, and the linear solve is
kept in the sheet's precision regardless so the physical gradient is unaffected.

Run::

    uv run python scripts/profile_memory.py
"""

from __future__ import annotations

import argparse
import math
import resource
import subprocess
import sys

import torch

from ddtwin.geometry import Sheet
from ddtwin.materials import MaterialField
from ddtwin.optimize import InverseDesignConfig, inverse_design
from ddtwin.physics import solve_static

# Mesh sizes to sweep: (rows, cols). N = rows * cols; the dense matrix is (3N, 3N).
SIZES = [(24, 40), (32, 56), (40, 72)]
DTYPE = torch.float32  # float32 so the dense-matrix bytes (and any AMP) are meaningful


def _peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB (macOS bytes, Linux kB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def _one_backward(rows: int, cols: int, checkpointed: bool) -> None:
    """One forward+backward of the compliance loss; used inside a worker process."""
    from torch.utils.checkpoint import checkpoint

    sheet = Sheet.cantilever(rows=rows, cols=cols, device="cpu", dtype=DTYPE)
    field = MaterialField(sheet, init=0.4)
    density = field.density()

    def loss_fn(d: torch.Tensor) -> torch.Tensor:
        result = solve_static(sheet, d)
        return result.compliance + 60.0 * torch.relu(d.mean() - 0.4) ** 2

    loss = checkpoint(loss_fn, density, use_reentrant=False) if checkpointed else loss_fn(density)
    loss.backward()
    print(f"{_peak_rss_mb():.1f}")


def _measure(rows: int, cols: int, checkpointed: bool) -> float:
    """Run one backward in a fresh subprocess and return its peak RSS in MB."""
    mode = "checkpoint" if checkpointed else "baseline"
    out = subprocess.run(
        [sys.executable, __file__, "--worker", str(rows), str(cols), mode],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip().splitlines()[-1])


def _dense_matrix_mb(n_nodes: int) -> float:
    """Bytes of the dense (3N, 3N) stiffness matrix, in MB (float32)."""
    n3 = 3 * n_nodes
    return (n3 * n3 * torch.finfo(DTYPE).bits // 8) / 1e6


def profile() -> None:
    print(f"device: CPU, dtype: {DTYPE}\n")
    print("Peak memory of one forward+backward, dense (3N,3N) solve:")
    print(
        f"{'nodes':>7} | {'3N':>6} | {'dense K (MB)':>12} | "
        f"{'baseline RSS':>13} | {'checkpoint RSS':>15} | {'saved':>7}"
    )
    print(f"{'-' * 7}-+-{'-' * 6}-+-{'-' * 12}-+-{'-' * 13}-+-{'-' * 15}-+-{'-' * 7}")
    for rows, cols in SIZES:
        n = rows * cols
        base = _measure(rows, cols, checkpointed=False)
        ckpt = _measure(rows, cols, checkpointed=True)
        saved = base - ckpt
        print(
            f"{n:>7} | {3 * n:>6} | {_dense_matrix_mb(n):>12.1f} | "
            f"{base:>10.1f} MB | {ckpt:>12.1f} MB | {saved:>4.0f} MB"
        )

    print(
        "\nReading: the peak tracks the dense (3N,3N) matrix, which grows O(N^2) and is\n"
        "what makes a high-resolution mesh run out of memory. Gradient checkpointing does\n"
        "NOT lower that peak here: the matrix and its factorisation are transient and are\n"
        "rebuilt in the checkpoint's backward recompute, so both passes hold the same\n"
        "O(N^2) object. Checkpointing pays off for deep or long-rollout models where many\n"
        "activations accumulate; this loop is a single static solve, so the real fix is a\n"
        "sparse stiffness assembly (the matrix is >99% zeros), which cuts the memory to\n"
        "O(N). The checkpoint / AMP plumbing is wired in and off by default."
    )


def verify() -> None:
    print("\nStability check: 5 iterations with checkpointing + AMP enabled")
    sheet = Sheet.cantilever(rows=24, cols=40, device="cpu", dtype=DTYPE)
    cfg = InverseDesignConfig(iters=5, seed=0, use_checkpoint=True, use_amp=True)
    result = inverse_design(sheet, cfg)
    losses = result.loss_history
    for i, val in enumerate(losses):
        print(f"  iter {i}: loss {val:.6f}")
    finite = all(math.isfinite(v) for v in losses)
    decreased = losses[-1] < losses[0]
    print(
        f"\n  all finite: {finite}   loss decreased: {decreased} "
        f"({losses[0]:.4f} -> {losses[-1]:.4f})"
    )
    if not (finite and decreased):
        raise SystemExit("stability check failed")
    print("  stability check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3, metavar=("ROWS", "COLS", "MODE"))
    args = parser.parse_args()

    if args.worker is not None:
        rows, cols, mode = args.worker
        _one_backward(int(rows), int(cols), checkpointed=(mode == "checkpoint"))
        return

    profile()
    verify()


if __name__ == "__main__":
    main()
