"""Round-trip of the web-player export format."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ddtwin.export import export_scene, load_positions
from ddtwin.geometry import Sheet
from ddtwin.optimize import InverseDesignConfig, inverse_design
from ddtwin.physics import SimConfig


def test_export_writes_consistent_files(tmp_path: Path) -> None:
    sheet = Sheet.cantilever(rows=5, cols=8, dtype=torch.float32)
    cfg = InverseDesignConfig(iters=10, sim=SimConfig(frames=60))
    res = inverse_design(sheet, cfg)
    export_scene(tmp_path, sheet, res, cfg)

    scene = json.loads((tmp_path / "scene.json").read_text())
    frames = scene["meta"]["frames"]
    nodes = scene["meta"]["nodes"]
    assert nodes == sheet.num_nodes
    assert len(scene["design"]["density"]) == nodes
    assert len(scene["topology"]["faces"]) == sheet.faces.numel()

    positions = load_positions(tmp_path / "positions.bin", frames, nodes)
    assert positions.shape == (frames, nodes, 3)
    # The exported final frame matches the simulated final frame.
    assert torch.allclose(positions[-1], res.final.positions[-1].to(torch.float32), atol=1e-5)
