"""
scenario_runner.py
==================
Integration function that connects:
  Reservoir outputs (flow rate, temperature, depth)
  → Surface system parameters (COP, efficiency)
  → LCOE model (openpyxl-based)
  → Financial outputs (LCoE €/GJ, heat/cooling MWth)

This is the single entry point that the scenario dashboard calls.
"""

import numpy as np
import pandas as pd
import openpyxl
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# Physical constants
RHO_WATER   = 1000.0   # kg/m³
CP_WATER    = 4.18     # kJ/(kg·K)
SECONDS_PER_HOUR = 3600.0
KW_TO_MW    = 1e-3


@dataclass
class ScenarioInputs:
    """All inputs required for one scenario run."""
    # Reservoir
    flow_rate_ls: float          # L/s (litres per second)
    prod_temp_c: float           # °C production temperature
    reinj_temp_c: float          # °C reinjection temperature
    well_depth_m: float          # m (for drilling cost)
    n_prod_wells: int = 1        # number of production wells
    n_inj_wells: int = 1         # number of injection wells

    # Surface system
    cop_heating: float = 4.0     # COP of heat pump for heating
    cop_cooling: float = 3.5     # EER of chiller for cooling
    fraction_for_cooling: float = 0.3  # fraction of thermal energy used for cooling

    # Economic
    lifetime_years: int = 30
    discount_rate: float = 0.07
    well_cost_eur_per_m: float = 3500.0   # €/m drilling cost
    surface_capex_eur: float = 3_000_000  # € for heat pump + surface equipment
    opex_fraction: float = 0.03           # annual OPEX as fraction of CAPEX

    # Labels
    scenario_name: str = "Base"


@dataclass
class ScenarioOutputs:
    """All computed outputs from one scenario run."""
    scenario_name: str
    # Thermal
    gross_thermal_kw: float      # kW geothermal heat extracted
    heat_output_mwth: float      # MWth delivered to district heating
    cooling_output_mwth: float   # MWth cooling delivered
    # Economic
    capex_eur: float
    annual_opex_eur: float
    lcoe_heating_eur_gj: float
    lcoe_cooling_eur_gj: float
    lcoe_blended_eur_gj: float
    npv_eur: float
    simple_payback_years: float
    # Flags
    meets_heat_target: bool      # >= 10 MWth
    meets_cooling_target: bool   # >= 5 MWth
    # Raw inputs (for reporting)
    inputs: ScenarioInputs = None


def compute_thermal_output(inputs: ScenarioInputs) -> dict:
    """
    Compute geothermal thermal extraction and delivered energy.

    Q_geo = flow [L/s] × density [kg/L] × Cp [kJ/(kg·K)] × ΔT [K]
    Q_delivered_heat = Q_geo × COP_hp / (COP_hp - 1)  [heat pump amplification]
    """
    flow_kg_s = inputs.flow_rate_ls * 1.0  # 1 kg/L for water
    delta_t = max(inputs.prod_temp_c - inputs.reinj_temp_c, 0.0)

    q_geo_kw = flow_kg_s * CP_WATER * delta_t   # kW geothermal extracted

    # Heat pump amplifies geothermal heat
    # Q_delivered = Q_geo * COP / (COP - 1)  (fraction going to heating)
    heat_fraction = 1.0 - inputs.fraction_for_cooling
    q_heat_kw = q_geo_kw * heat_fraction * inputs.cop_heating / (inputs.cop_heating - 1)
    q_cool_kw = q_geo_kw * inputs.fraction_for_cooling * inputs.cop_cooling

    return {
        "q_geo_kw": round(q_geo_kw, 1),
        "q_heat_mwth": round(q_heat_kw * KW_TO_MW, 3),
        "q_cool_mwth": round(q_cool_kw * KW_TO_MW, 3),
    }


def compute_capex(inputs: ScenarioInputs) -> float:
    """
    Total CAPEX = drilling costs + surface equipment.
    Drilling = n_wells × depth × cost_per_metre
    """
    n_wells = inputs.n_prod_wells + inputs.n_inj_wells
    drilling_cost = n_wells * inputs.well_depth_m * inputs.well_cost_eur_per_m
    total_capex = drilling_cost + inputs.surface_capex_eur
    return total_capex


def compute_lcoe(annual_energy_gj: float, capex_eur: float,
                 annual_opex_eur: float, lifetime: int,
                 discount_rate: float) -> float:
    """
    Levelized Cost of Energy (€/GJ).

    LCoE = (CAPEX × CRF + annual_OPEX) / annual_energy_GJ

    CRF = Capital Recovery Factor = r(1+r)^n / ((1+r)^n - 1)
    """
    if annual_energy_gj <= 0:
        return np.inf

    r = discount_rate
    n = lifetime
    crf = (r * (1 + r)**n) / ((1 + r)**n - 1)

    annual_capital_cost = capex_eur * crf
    annual_total_cost = annual_capital_cost + annual_opex_eur
    lcoe = annual_total_cost / annual_energy_gj
    return lcoe


def compute_npv(annual_revenue_eur: float, capex_eur: float,
                annual_opex_eur: float, lifetime: int,
                discount_rate: float) -> float:
    """Simple NPV calculation assuming constant annual net cash flow."""
    annual_cashflow = annual_revenue_eur - annual_opex_eur
    r = discount_rate
    n = lifetime
    pv_cashflows = annual_cashflow * ((1 - (1 + r)**-n) / r)
    npv = pv_cashflows - capex_eur
    return npv


def run_scenario(inputs: ScenarioInputs,
                 heat_price_eur_gj: float = 20.0,
                 cooling_price_eur_gj: float = 15.0) -> ScenarioOutputs:
    """
    Main integration function. Run a complete geothermal scenario.

    Parameters
    ----------
    inputs : ScenarioInputs
        All reservoir, surface, and economic parameters
    heat_price_eur_gj : float
        Market price of heat in €/GJ (used for NPV / revenue calculation)
    cooling_price_eur_gj : float
        Market price of cooling in €/GJ

    Returns
    -------
    ScenarioOutputs
        Complete set of computed results
    """
    # --- Thermal ---
    thermal = compute_thermal_output(inputs)
    q_heat_mwth  = thermal["q_heat_mwth"]
    q_cool_mwth  = thermal["q_cool_mwth"]
    q_geo_kw     = thermal["q_geo_kw"]

    # Convert MW to GJ/year (8760 hours/year, assuming 95% availability)
    availability = 0.95
    hours_per_year = 8760.0 * availability

    heat_gj_yr   = q_heat_mwth  * 1000.0 * 3600.0 * hours_per_year / 1e6
    cool_gj_yr   = q_cool_mwth  * 1000.0 * 3600.0 * hours_per_year / 1e6

    # --- Economic ---
    capex = compute_capex(inputs)
    annual_opex = capex * inputs.opex_fraction

    lcoe_heat   = compute_lcoe(heat_gj_yr, capex, annual_opex,
                               inputs.lifetime_years, inputs.discount_rate)
    lcoe_cool   = compute_lcoe(cool_gj_yr, capex, annual_opex,
                               inputs.lifetime_years, inputs.discount_rate)

    total_gj_yr = heat_gj_yr + cool_gj_yr
    lcoe_blend  = compute_lcoe(total_gj_yr, capex, annual_opex,
                               inputs.lifetime_years, inputs.discount_rate)

    annual_revenue = (heat_gj_yr * heat_price_eur_gj +
                      cool_gj_yr * cooling_price_eur_gj)

    npv = compute_npv(annual_revenue, capex, annual_opex,
                      inputs.lifetime_years, inputs.discount_rate)

    payback = capex / max(annual_revenue - annual_opex, 1.0)

    return ScenarioOutputs(
        scenario_name       = inputs.scenario_name,
        gross_thermal_kw    = q_geo_kw,
        heat_output_mwth    = q_heat_mwth,
        cooling_output_mwth = q_cool_mwth,
        capex_eur           = round(capex, 0),
        annual_opex_eur     = round(annual_opex, 0),
        lcoe_heating_eur_gj = round(lcoe_heat, 2),
        lcoe_cooling_eur_gj = round(lcoe_cool, 2),
        lcoe_blended_eur_gj = round(lcoe_blend, 2),
        npv_eur             = round(npv, 0),
        simple_payback_years= round(payback, 1),
        meets_heat_target   = q_heat_mwth >= 10.0,
        meets_cooling_target= q_cool_mwth >= 5.0,
        inputs              = inputs,
    )


def run_standard_scenarios(thermogis_row: dict) -> pd.DataFrame:
    """
    Run P90 (low), P50 (base), P10 (high) scenarios for a given well.

    Parameters
    ----------
    thermogis_row : dict
        Dictionary with P90/P50/P10 flow rates, temperatures, depths
        Keys expected: flow_p90, flow_p50, flow_p10,
                       temp_p90, temp_p50, temp_p10,
                       depth_p90, depth_p50, depth_p10

    Returns
    -------
    pd.DataFrame with one row per scenario
    """
    scenarios_def = [
        ("P90 (Low)",  "p90"),
        ("P50 (Base)", "p50"),
        ("P10 (High)", "p10"),
    ]

    results = []
    for name, suffix in scenarios_def:
        inp = ScenarioInputs(
            scenario_name      = name,
            flow_rate_ls       = thermogis_row.get(f"flow_{suffix}", 10.0) / 3.6,  # m³/h → L/s
            prod_temp_c        = thermogis_row.get(f"temp_{suffix}", 70.0),
            reinj_temp_c       = 40.0,
            well_depth_m       = thermogis_row.get(f"depth_{suffix}", 2000.0),
            cop_heating        = 4.0,
            cop_cooling        = 3.5,
            fraction_for_cooling = 0.3,
        )
        out = run_scenario(inp)
        results.append({
            "Scenario":             out.scenario_name,
            "Flow Rate (m³/h)":     inp.flow_rate_ls * 3.6,
            "Prod Temp (°C)":       inp.prod_temp_c,
            "Heat Output (MWth)":   out.heat_output_mwth,
            "Cooling Output (MWth)":out.cooling_output_mwth,
            "CAPEX (M€)":           round(out.capex_eur / 1e6, 2),
            "LCoE Heat (€/GJ)":     out.lcoe_heating_eur_gj,
            "LCoE Blended (€/GJ)":  out.lcoe_blended_eur_gj,
            "NPV (M€)":             round(out.npv_eur / 1e6, 2),
            "Payback (yrs)":        out.simple_payback_years,
            "Meets Heat Target":    out.meets_heat_target,
            "Meets Cooling Target": out.meets_cooling_target,
        })

    return pd.DataFrame(results)
