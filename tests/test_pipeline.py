"""
test_pipeline.py
================
Basic sanity tests for the AGD 2026 data pipeline.
Run with: python -m pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.las_loader import load_las, load_all_wells, get_curve_coverage
from src.pipeline.depth_converter import add_tvd_to_vertical_well
from src.models.scenario_runner import run_scenario, ScenarioInputs
from src.models.feasibility import FeasibilityScorer, WellInputs


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
EXTERNAL_DIR = Path(__file__).parent.parent / "data" / "external"


# ───── LAS Loader Tests ─────

class TestLASLoader:

    def test_load_single_las(self):
        """BLT-01 should load without errors."""
        las_path = RAW_DIR / "BLT-01.las"
        if not las_path.exists():
            pytest.skip("BLT-01.las not found")
        df = load_las(las_path, well_name="BLT-01")
        assert "WELL_ID" in df.columns
        assert "DEPTH_M" in df.columns
        assert df["WELL_ID"].iloc[0] == "BLT-01"
        assert len(df) > 0

    def test_null_replacement(self):
        """Null values (-999.25) must be replaced with np.nan."""
        las_path = RAW_DIR / "BLT-01.las"
        if not las_path.exists():
            pytest.skip("BLT-01.las not found")
        df = load_las(las_path)
        for col in df.select_dtypes(include=[np.number]).columns:
            assert not (df[col] == -999.25).any(), \
                f"Column {col} still contains -999.25 sentinel values"

    def test_load_all_wells(self):
        """All 4 wells should load."""
        if not RAW_DIR.exists():
            pytest.skip("Raw data directory not found")
        wells = load_all_wells(RAW_DIR)
        assert len(wells) == 4
        assert "BLT-01" in wells
        assert "JUT-01" in wells

    def test_curve_coverage(self):
        """Coverage table should have wells as index."""
        if not RAW_DIR.exists():
            pytest.skip("Raw data directory not found")
        wells = load_all_wells(RAW_DIR)
        coverage = get_curve_coverage(wells)
        assert set(coverage.index) == set(wells.keys())
        assert all(0 <= v <= 100 for v in coverage.values.flatten()
                   if not np.isnan(v))


# ───── Depth Converter Tests ─────

class TestDepthConverter:

    def test_vertical_well_tvd_equals_md(self):
        """For vertical wells, TVD should equal MD."""
        df = pd.DataFrame({
            "WELL_ID": ["BLT-01"] * 5,
            "DEPTH_M": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        })
        result = add_tvd_to_vertical_well(df)
        assert "TVD_M" in result.columns
        np.testing.assert_array_almost_equal(
            result["DEPTH_M"].values, result["TVD_M"].values
        )


# ───── Scenario Runner Tests ─────

class TestScenarioRunner:

    def test_base_scenario_runs(self):
        """A valid scenario should return ScenarioOutputs without error."""
        inp = ScenarioInputs(
            scenario_name="Test Base",
            flow_rate_ls=30.0,      # L/s = ~108 m³/h
            prod_temp_c=80.0,
            reinj_temp_c=40.0,
            well_depth_m=2000.0,
            cop_heating=4.0,
            cop_cooling=3.5,
        )
        out = run_scenario(inp)
        assert out.heat_output_mwth > 0
        assert out.cooling_output_mwth > 0
        assert out.capex_eur > 0
        assert out.lcoe_blended_eur_gj > 0

    def test_zero_flow_returns_inf_lcoe(self):
        """Zero flow rate should return infinite LCoE (no energy produced)."""
        inp = ScenarioInputs(
            scenario_name="Zero flow",
            flow_rate_ls=0.0,
            prod_temp_c=80.0,
            reinj_temp_c=40.0,
            well_depth_m=2000.0,
        )
        out = run_scenario(inp)
        assert out.heat_output_mwth == 0.0
        assert out.lcoe_heating_eur_gj == float("inf")

    def test_heat_target_flag(self):
        """High flow rate should meet 10 MWth target."""
        inp = ScenarioInputs(
            scenario_name="High flow",
            flow_rate_ls=60.0,   # 216 m³/h — should easily meet target
            prod_temp_c=90.0,
            reinj_temp_c=35.0,
            well_depth_m=2200.0,
            cop_heating=4.5,
            fraction_for_cooling=0.2,
        )
        out = run_scenario(inp)
        assert out.meets_heat_target, \
            f"Expected heat target met, got {out.heat_output_mwth} MWth"


# ───── Feasibility Scorer Tests ─────

class TestFeasibilityScorer:

    def test_score_range(self):
        """Scores must always be in 0–100."""
        scorer = FeasibilityScorer()
        test_cases = [
            WellInputs("TEST-A", "P50", 100.0, 80.0, 100.0, 150.0, 2.0, 2000.0),
            WellInputs("TEST-B", "P50", 5.0,   55.0, 1.0,   20.0,  8.0, 3200.0),
            WellInputs("TEST-C", "P50", 0.0,   50.0, 0.0,   0.0,   10.0, 3500.0),
        ]
        for inp in test_cases:
            result = scorer.score_well(inp)
            assert 0 <= result["score_total"] <= 100, \
                f"Score out of range for {inp.well_id}: {result['score_total']}"

    def test_better_well_scores_higher(self):
        """A well with better flow/temp/perm should score higher."""
        scorer = FeasibilityScorer()
        good = WellInputs("GOOD", "P50", 150.0, 90.0, 150.0, 200.0, 1.0, 1800.0)
        poor = WellInputs("POOR", "P50", 10.0,  55.0, 2.0,   30.0,  9.0, 3400.0)
        score_good = scorer.score_well(good)["score_total"]
        score_poor = scorer.score_well(poor)["score_total"]
        assert score_good > score_poor, \
            f"Good well ({score_good}) should outscore poor well ({score_poor})"

    def test_weights_sum_to_one(self):
        """Scoring weights must sum to 1.0."""
        from src.models.feasibility import ScoringWeights
        w = ScoringWeights()
        assert abs(sum(vars(w).values()) - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
