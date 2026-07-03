"""Export an optimised result to the web player's data format.

Two files are written into a target directory (the player loads both):

* ``positions.bin`` -- raw little-endian ``float32`` of shape ``(frames, N, 3)``, the node
  trajectory. Binary keeps it ~6x smaller than JSON and parses instantly into a typed array,
  mirroring the binary-protocol convention used across the rest of the portfolio.
* ``scene.json`` -- topology and metadata: grid shape, render faces, spring edges, the
  optimised density field (for the material colour map), boundary conditions, the load
  vector, bounds, and the optimisation history.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ddtwin.geometry import Sheet
from ddtwin.optimize import InverseDesignConfig, InverseDesignResult


def export_scene(
    out_dir: str | Path,
    sheet: Sheet,
    result: InverseDesignResult,
    cfg: InverseDesignConfig,
) -> Path:
    """Write ``positions.bin`` and ``scene.json`` into ``out_dir``; return the directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    positions = result.final.positions.detach().to(torch.float32).cpu().contiguous()
    frames, nodes, _ = positions.shape
    (out / "positions.bin").write_bytes(positions.numpy().tobytes())

    rest = sheet.rest.detach().to(torch.float32).cpu()
    bounds_min = positions.reshape(-1, 3).min(dim=0).values.tolist()
    bounds_max = positions.reshape(-1, 3).max(dim=0).values.tolist()

    scene = {
        "meta": {
            "generator": "ddtwin",
            "rows": sheet.rows,
            "cols": sheet.cols,
            "nodes": nodes,
            "frames": frames,
            "dtype": "float32",
            "layout": "frames,nodes,xyz",
        },
        "topology": {
            "faces": sheet.faces.cpu().flatten().tolist(),
            "edges": sheet.edges.cpu().flatten().tolist(),
            "rest": rest.flatten().tolist(),
        },
        "boundary": {
            "pinned": sheet.pinned.cpu().to(torch.uint8).tolist(),
            "loaded": sheet.loaded.cpu().to(torch.uint8).tolist(),
            "load_vector": list(cfg.sim.load),
        },
        "design": {
            "density": result.density.detach().to(torch.float32).cpu().tolist(),
            "volume_fraction": cfg.vol_frac,
            "compliance": result.compliance,
            "uniform_compliance": result.uniform_compliance,
            "improvement": result.improvement,
            "tip_deflection": result.tip_deflection,
            "uniform_deflection": result.uniform_tip_deflection,
            "deflection_improvement": result.deflection_improvement,
        },
        "history": {
            "loss": result.loss_history,
            "compliance": result.compliance_history,
            "volume": result.volume_history,
        },
        "bounds": {"min": bounds_min, "max": bounds_max},
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=2))
    return out


def load_positions(path: str | Path, frames: int, nodes: int) -> torch.Tensor:
    """Re-read ``positions.bin`` (used by tests / round-trip checks)."""
    raw = Path(path).read_bytes()
    flat = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    return flat.reshape(frames, nodes, 3)
