"""Topology and boundary-condition sanity checks for the cantilever sheet."""

from __future__ import annotations

import pytest
import torch

from ddtwin.geometry import KIND_BEND, KIND_SHEAR, KIND_STRUCT, Sheet


def test_node_and_edge_counts() -> None:
    sheet = Sheet.cantilever(rows=5, cols=7)
    assert sheet.num_nodes == 35
    # Two triangles per grid cell.
    assert sheet.faces.shape == ((5 - 1) * (7 - 1) * 2, 3)
    # Every spring family is present.
    kinds = set(sheet.edge_kind.tolist())
    assert kinds == {KIND_STRUCT, KIND_SHEAR, KIND_BEND}


def test_edges_reference_valid_nodes() -> None:
    sheet = Sheet.cantilever(rows=6, cols=9)
    assert int(sheet.edges.min()) >= 0
    assert int(sheet.edges.max()) < sheet.num_nodes
    # No self-loops.
    assert torch.all(sheet.edges[:, 0] != sheet.edges[:, 1])


def test_rest_lengths_positive() -> None:
    sheet = Sheet.cantilever(rows=4, cols=8, spacing=0.5)
    assert torch.all(sheet.rest_length > 0)


def test_boundary_conditions() -> None:
    sheet = Sheet.cantilever(rows=5, cols=6, load_columns=1)
    # Clamp on the j == 0 edge: exactly `rows` pinned nodes.
    assert int(sheet.pinned.sum()) == 5
    # Load on the j == cols-1 edge.
    assert int(sheet.loaded.sum()) == 5
    # Clamped and loaded sets are disjoint.
    assert not bool((sheet.pinned & sheet.loaded).any())


def test_requires_minimum_size() -> None:
    with pytest.raises(ValueError):
        Sheet.cantilever(rows=2, cols=2)
