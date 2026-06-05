"""
feasibility.py
==============
Composite geothermal feasibility scoring system.

Takes reservoir properties per well (from ThermoGIS + ML models)
and produces a 0–100 score under P10 / P50 / P90 scenarios.

Score components (weights add to 100):
  - Flow rate (m³/h)      : 30 pts  — directly drives thermal output
  - Temperature (°C)      : 25 pts  — drives heat content
  - Permeability (mD)     : 20 pts  — reservoir deliverability
  - Thickness (m)         : 10 pts  — reservoir volume
  - Distance to USP (km)  : 10 pts  — pipeline cost proxy (closer = better)
  - Depth (m)             : 5 pts   — drilling cost proxy (shallower = better)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScoringWeights:
    flow_rate:   float = 0.30
    temperature: float = 0.25
    permeability: float = 0.20
    thickness:   float = 0.10
    distance:    float = 0.10
    depth:       float = 0.05

    def validate(self):
        total = sum(vars(self).values())
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"


# Reference ranges for normalisation (based on ThermoGIS data + literature)
REFERENCE_RANGES = {
    # parameter: (min_value, max_value)  — normalised to 0-1 in this range
    "flow_rate_m3h":   (0.0,   200.0),   # m³/h
    "temperature_c":   (50.0,  120.0),   # °C
    "permeability_md": (0.0,   200.0),   # mD
    "thickness_m":     (0.0,   300.0),   # m
    "distance_km":     (0.0,   10.0),    # km (inverted — closer is better)
    "depth_m":         (1000.0, 3500.0), # m  (inverted — shallower is better)
}


def _normalise(value: float, param: str, invert: bool = False) -> float:
    """
    Normalise a value to 0–1 based on reference range.

    Parameters
    ----------
    value : float
    param : str
        Key in REFERENCE_RANGES
    invert : bool
        If True, lower value → higher score (used for distance and depth)

    Returns
    -------
    float in [0, 1]
    """
    lo, hi = REFERENCE_RANGES[param]
    if hi == lo:
        return 0.5
    norm = (value - lo) / (hi - lo)
    norm = float(np.clip(norm, 0.0, 1.0))
    return (1.0 - norm) if invert else norm


@dataclass
class WellInputs:
    """Input parameters for one well under one scenario (P10/P50/P90)."""
    well_id: str
    scenario: str                    # "P10", "P50", or "P90"
    flow_rate_m3h: float             # m³/hour
    temperature_c: float             # °C
    permeability_md: float           # mD (mean over Rotliegend)
    thickness_m: float               # net reservoir thickness (m)
    distance_to_usp_km: float        # km to urban service point
    depth_m: float                   # mean Rotliegend depth (m TVD)
    heat_output_mwth: Optional[float] = None   # derived
    notes: str = ""


class FeasibilityScorer:
    """
    Scores geothermal well candidates on a 0–100 scale.

    Usage
    -----
    scorer = FeasibilityScorer()
    results = scorer.score_all(well_inputs_list)
    scorer.print_summary(results)
    """

    THRESHOLD_VIABLE = 50.0    # score >= 50 → viable
    THRESHOLD_STRONG = 70.0    # score >= 70 → strongly viable

    def __init__(self, weights: ScoringWeights = None):
        self.weights = weights or ScoringWeights()
        self.weights.validate()

    def score_well(self, inputs: WellInputs) -> dict:
        """
        Score a single well under a single scenario.

        Returns
        -------
        dict with component scores and total score
        """
        w = self.weights

        # Normalise each parameter
        n_flow  = _normalise(inputs.flow_rate_m3h,    "flow_rate_m3h")
        n_temp  = _normalise(inputs.temperature_c,    "temperature_c")
        n_perm  = _normalise(inputs.permeability_md,  "permeability_md")
        n_thick = _normalise(inputs.thickness_m,      "thickness_m")
        n_dist  = _normalise(inputs.distance_to_usp_km, "distance_km", invert=True)
        n_depth = _normalise(inputs.depth_m,          "depth_m",       invert=True)

        # Weighted composite score (0–100)
        total = 100.0 * (
            w.flow_rate    * n_flow  +
            w.temperature  * n_temp  +
            w.permeability * n_perm  +
            w.thickness    * n_thick +
            w.distance     * n_dist  +
            w.depth        * n_depth
        )

        # Compute thermal output estimate
        # Q_th = flow_rate [m³/h] × density [kg/m³] × Cp [kJ/kg·K] × ΔT [K] / 3600
        # Assume ΔT = production_temp - 40°C (reinjection), density=1000, Cp=4.18
        delta_t = max(inputs.temperature_c - 40.0, 0.0)
        q_th_mwth = (inputs.flow_rate_m3h * 1000.0 * 4.18 * delta_t) / (3600.0 * 1000.0)

        verdict = (
            "STRONG"   if total >= self.THRESHOLD_STRONG  else
            "VIABLE"   if total >= self.THRESHOLD_VIABLE  else
            "MARGINAL"
        )

        return {
            "well_id":          inputs.well_id,
            "scenario":         inputs.scenario,
            "score_total":      round(total, 1),
            "verdict":          verdict,
            "score_flow":       round(n_flow  * w.flow_rate    * 100, 1),
            "score_temp":       round(n_temp  * w.temperature  * 100, 1),
            "score_perm":       round(n_perm  * w.permeability * 100, 1),
            "score_thick":      round(n_thick * w.thickness    * 100, 1),
            "score_dist":       round(n_dist  * w.distance     * 100, 1),
            "score_depth":      round(n_depth * w.depth        * 100, 1),
            "flow_rate_m3h":    inputs.flow_rate_m3h,
            "temperature_c":    inputs.temperature_c,
            "permeability_md":  inputs.permeability_md,
            "thickness_m":      inputs.thickness_m,
            "distance_km":      inputs.distance_to_usp_km,
            "depth_m":          inputs.depth_m,
            "heat_output_mwth": round(q_th_mwth, 2),
            "meets_heat_target": q_th_mwth >= 10.0,
            "notes":            inputs.notes,
        }

    def score_all(self, inputs_list: list[WellInputs]) -> pd.DataFrame:
        """
        Score multiple wells / scenarios.

        Returns
        -------
        pd.DataFrame sorted by scenario then score descending
        """
        records = [self.score_well(inp) for inp in inputs_list]
        df = pd.DataFrame(records)
        df = df.sort_values(["scenario", "score_total"], ascending=[True, False])
        df = df.reset_index(drop=True)
        return df

    def print_summary(self, results_df: pd.DataFrame):
        """Print a formatted summary table."""
        print("\n" + "="*80)
        print("GEOTHERMAL FEASIBILITY SCORES")
        print("="*80)
        cols = ["well_id", "scenario", "score_total", "verdict",
                "heat_output_mwth", "meets_heat_target",
                "flow_rate_m3h", "temperature_c", "permeability_md"]
        print(results_df[cols].to_string(index=False))
        print("="*80)

        best = results_df.loc[results_df["score_total"].idxmax()]
        print(f"\nTop candidate: {best['well_id']} ({best['scenario']}) "
              f"— Score {best['score_total']}/100 — {best['verdict']}")
        print(f"Estimated heat output: {best['heat_output_mwth']} MWth")
        if best["meets_heat_target"]:
            print("✓ Meets 10 MWth heating target")
        else:
            print("✗ Does NOT meet 10 MWth heating target at this scenario")


def build_inputs_from_thermogis(thermogis_df: pd.DataFrame) -> list[WellInputs]:
    """
    Build WellInputs objects from ThermoGIS DataFrame.
    Expects columns with P90/P50/P10 suffixes for flow, temp, perm, thickness, depth.

    Returns list ready for FeasibilityScorer.score_all()
    """
    inputs = []
    scenario_map = {"P90": "P90", "P50": "P50", "P10": "P10"}

    # Column detection — ThermoGIS uses various naming conventions
    col_map = {}
    for col in thermogis_df.columns:
        cu = col.upper()
        if "FLOW" in cu or "RATE" in cu:     col_map["flow"]  = col
        if "TEMP" in cu or "TEMP_C" in cu:   col_map["temp"]  = col
        if "PERM" in cu or "_K_" in cu:      col_map["perm"]  = col
        if "THICK" in cu or "NET" in cu:     col_map["thick"] = col
        if "DEPTH" in cu:                    col_map["depth"] = col
        if "DIST" in cu or "USP" in cu:      col_map["dist"]  = col

    print(f"ThermoGIS column mapping: {col_map}")

    for _, row in thermogis_df.iterrows():
        well = str(row.get("WELL_ID", row.get("WELL", "UNKNOWN")))
        for suffix, scenario in scenario_map.items():
            try:
                inp = WellInputs(
                    well_id=well,
                    scenario=scenario,
                    flow_rate_m3h=float(row.get(f"FLOW_{suffix}", row.get("FLOW_RATE_M3H", 0))),
                    temperature_c=float(row.get(f"TEMP_{suffix}", row.get("TEMPERATURE_C", 60))),
                    permeability_md=float(row.get(f"PERM_{suffix}", row.get("PERM_MD", 1))),
                    thickness_m=float(row.get(f"THICK_{suffix}", row.get("THICKNESS_M", 100))),
                    distance_to_usp_km=float(row.get("DIST_USP_KM", row.get("DISTANCE_KM", 3))),
                    depth_m=float(row.get(f"DEPTH_{suffix}", row.get("DEPTH_M", 2000))),
                )
                inputs.append(inp)
            except (ValueError, TypeError) as e:
                print(f"  Warning: Could not build {well} {scenario} inputs — {e}")

    return inputs
