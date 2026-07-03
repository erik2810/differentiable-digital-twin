"""Differentiable Digital Twin.

A compact, fully differentiable mass-spring structural solver and a gradient-based
inverse-design loop that distributes a fixed material budget across a deformable
sheet so it best resists a prescribed external load. The optimised trajectory is
exported for a WebGPU browser player.

The public surface is intentionally small:

    from ddtwin import Sheet, SimConfig, solve_static, simulate, inverse_design

See ``ddtwin.cli`` for the command-line entry point.
"""

from __future__ import annotations

from ddtwin.geometry import Sheet
from ddtwin.materials import MaterialField
from ddtwin.optimize import InverseDesignConfig, InverseDesignResult, inverse_design
from ddtwin.physics import SimConfig, SimResult, StaticResult, simulate, solve_static

__all__ = [
    "InverseDesignConfig",
    "InverseDesignResult",
    "MaterialField",
    "Sheet",
    "SimConfig",
    "SimResult",
    "StaticResult",
    "inverse_design",
    "simulate",
    "solve_static",
]

__version__ = "0.1.0"
