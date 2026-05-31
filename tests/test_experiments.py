"""Tests for experiment aggregation and material probing."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.aggregate_all_results import discover_reports, load_report_json


def test_discover_reports_finds_v1_512():
    root = Path("results")
    if not (root / "MetalPlates013" / "v1-512" / "report.json").exists():
        return  # skip if no real runs in workspace
    paths = discover_reports(root)
    assert any("v1-512" in str(p) for p in paths)


def test_load_report_json_shape():
    path = Path("results/MetalPlates013/v1-512/report.json")
    if not path.exists():
        return
    row = load_report_json(path)
    assert row is not None
    assert row["material"] == "MetalPlates013"
    assert "avg_psnr_baseline" in row
    assert "storage_ratio" in row


def test_material_inventory_exists():
    inv = Path("results/experiments/material_inventory.json")
    if not inv.exists():
        return
    data = json.loads(inv.read_text())
    assert "materials" in data
    assert len(data["materials"]) >= 5
