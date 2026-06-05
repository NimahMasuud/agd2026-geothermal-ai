# AGD 2026 — AI Workflow Engineer Codebase

**SPE Africa Geothermal Datathon 2026**
**Role:** Team Lead & AI Workflow Engineer
**Submission deadline:** June 4, 2026

---

## Project Overview

This repository implements the complete AI engineering workflow for assessing
geothermal potential of Rotliegend sandstone wells near Utrecht, Netherlands.

**Core question:** Can well(s) in the dataset supply ≥10 MWth heating and ≥5 MWth
cooling to a Utrecht urban district?

---

## Quickstart

```bash
# 1. Clone and enter the repo
cd agd2026-team

# 2. Create environment (choose one)
pip install -r requirements.txt
# OR
conda env create -f environment.yml && conda activate agd2026

# 3. Set your Anthropic API key (for notebook 08 bonus challenge)
export ANTHROPIC_API_KEY="your-key-here"

# 4. Run notebooks in order
jupyter notebook notebooks/
```

---

## Repository Structure

```
agd2026-team/
├── data/
│   ├── raw/              ← Original LAS well log files (read-only)
│   ├── external/         ← ThermoGIS, lithostratigraphy, well path data
│   ├── economic/         ← LCOE.xlsx model
│   ├── processed/        ← Generated: cleaned unified DataFrames
│   └── interim/          ← Generated: imputed outputs
├── notebooks/
│   ├── 01_data_exploration.ipynb      ← LAS loading, EDA, coverage map
│   ├── 02_data_cleaning.ipynb         ← Null handling, JUT-01 TVD, schema
│   ├── 03_ml_imputation.ipynb         ← XGBoost quantile imputation
│   ├── 04_permeability_model.ipynb    ← Perm prediction + SHAP
│   ├── 05_feasibility_score.ipynb     ← Composite well scoring
│   ├── 06_integration.ipynb           ← Reservoir → surface → LCOE
│   ├── 07_scenario_dashboard.ipynb    ← Plotly scenario comparison
│   └── 08_ai_workflow_bonus.ipynb     ← GeothermalAIAssistant demo
├── src/
│   ├── pipeline/
│   │   ├── las_loader.py        ← LAS parsing, null handling
│   │   ├── depth_converter.py   ← MD→TVD for JUT-01 (deviated well)
│   │   └── schema_builder.py    ← Unified DataFrame builder + formation tags
│   ├── models/
│   │   ├── imputer.py           ← LogImputer (quantile XGBoost)
│   │   ├── permeability.py      ← PermeabilityModel (XGBoost + SHAP)
│   │   ├── feasibility.py       ← FeasibilityScorer (0–100 composite)
│   │   └── scenario_runner.py   ← run_scenario() integration function
│   └── ai_workflow/
│       └── assistant.py         ← GeothermalAIAssistant (Anthropic API)
├── outputs/
│   ├── figures/          ← All saved plots
│   ├── models/           ← Saved .pkl model files
│   └── reports/          ← Scenario tables, AI assessment reports
├── tests/
│   └── test_pipeline.py  ← pytest test suite
├── config/
│   └── settings.yaml     ← All paths and hyperparameters
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Wells in Dataset

| Well    | Type     | Notes                                     |
|---------|----------|-------------------------------------------|
| BLT-01  | Vertical | Most complete logs; best ThermoGIS values |
| EVD-01  | Vertical | Sparse curves (GR, RHOB, DT only)         |
| JUT-01  | Deviated | Requires MD→TVD conversion; moderate res. |
| PKP-01  | Vertical | Moderate logs; lower perm estimates       |

---

## Key Source Files

| Module               | What it does                                      |
|----------------------|---------------------------------------------------|
| `las_loader.py`      | Loads all 4 LAS files, strips nulls               |
| `depth_converter.py` | Converts JUT-01 measured depth → true vertical    |
| `schema_builder.py`  | Merges all wells into unified DataFrame           |
| `imputer.py`         | Fills missing NPHI, DT, RS via XGBoost P10/P50/P90|
| `permeability.py`    | Predicts depth-resolved perm from log data        |
| `feasibility.py`     | Scores each well 0–100 across scenarios           |
| `scenario_runner.py` | Full integration: reservoir → surface → LCOE      |
| `assistant.py`       | AI assessment reports via Anthropic API           |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Notebook Execution Order

Run notebooks **01 → 08** in sequence. Each notebook saves its output to
`data/processed/` or `data/interim/` which the next notebook reads.

| Notebook | Reads                          | Writes                              |
|----------|--------------------------------|-------------------------------------|
| 01       | data/raw/*.las                 | (in-memory EDA only)                |
| 02       | raw LAS + external xlsx        | data/processed/all_wells_merged.csv |
| 03       | all_wells_merged.csv           | data/interim/all_wells_imputed.csv  |
| 04       | all_wells_imputed.csv          | data/processed/permeability_profiles.csv |
| 05       | permeability_profiles.csv + ThermoGIS | outputs/reports/feasibility_scores.csv |
| 06       | feasibility_scores.csv + LCOE.xlsx | outputs/reports/scenario_results.csv |
| 07       | scenario_results.csv           | outputs/reports/scenario_dashboard.html |
| 08       | All processed data             | outputs/reports/*_assessment.md     |

---

## Contact
Team Lead & AI Workflow Engineer: Neemah
Competition: SPE Africa Geothermal Datathon 2026
Email: speafricageodatathon@gmail.com
