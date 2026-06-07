# SPE Africa Geothermal Datathon 2026: Techno-Economic & AI Workflow Assessment

**Team Name:** Thermal Edge Geo AI

**Role:** Team Lead & AI Workflow Engineer

**Submission Deadline:** June 7, 2026

---

## Project Overview

This repository implements a production-grade AI engineering workflow to evaluate the deep geothermal heating and cooling potential of the Rotliegend Sandstone formations near Utrecht, Netherlands.

### Core Engineering Requirements

- **Heating Capacity Target:** Deliver a sustained thermal throughput of ≥10 MWth to an urban district
- **Cooling Capacity Target:** Deliver a sustained cooling throughput of ≥5 MWth

### Primary Asset Determination

- **Selected Candidate:** BLT-01 is identified as the primary viable asset
- **Feasibility Evaluation:** Achieved a composite score of 62.9/100 (VIABLE)
- **Production Profile:** Delivers an estimated thermal output of 20 MWth at P50 with an operational flow rate of 105 m³/h

---

## Key Results

| Well   | Score | Verdict              | Heat Output | Recommendation       |
|--------|-------|----------------------|-------------|----------------------|
| BLT-01 | 62.9  | VIABLE               | 20 MWth     | ✅ DEVELOP            |
| EVD-01 | 40.9  | MARGINAL             | —           | 🔍 INVESTIGATE FURTHER|
| JUT-01 | 31.1  | MARGINAL             | —           | 🔍 INVESTIGATE FURTHER|
| PKP-01 | 22.7  | MARGINAL             | —           | 🔍 INVESTIGATE FURTHER|

---

## Technical Report

The full technical report is included in this repository as `ThermalEdge_Technical_Report.docx`. It covers the complete project narrative including all methodology, results and recommendations across eight sections:

1. **Executive Summary** — strategic motivation and system overview
2. **Methodology and Assumptions** — geothermal gradient, COP, recovery factor
3. **External Data and GIS Mapping** — ThermoGIS and OpenStreetMap spatial analysis
4. **Reservoir Assessment (Challenge 1)** — preliminary and AI-optimised well evaluation for BLT-01, EVD-01, JUT-01 and PKP-01
5. **Integrated Surface System Design (Challenge 2)** — heat pumps, absorption chillers, TES sizing and energy balance
6. **AI-Assisted Workflow (Bonus Challenge)** — XGBoost imputation, feasibility scoring and LLM-generated well reports
7. **Economic Evaluation** — LCoE of €28.53/MWhth, CAPEX €11.72M, sensitivity and Tornado analysis
8. **Conclusion and Recommendations** — BLT-01 confirmed as primary development candidate

---
## Quickstart

### Step 1 — Clone and enter the repo
```bash
git clone https://github.com/NimahMasuud/agd2026-geothermal-ai.git
cd agd2026-geothermal-ai
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Set your AI API key (for notebook 08 bonus challenge)

The assistant **auto-detects** which key you have set. Choose ONE option:

**Option A — Groq (FREE Recommended — https://console.groq.com)**

Windows PowerShell:
```powershell
$env:GROQ_API_KEY = "gsk_your-groq-key-here"
```
Mac/Linux:
```bash
export GROQ_API_KEY="gsk_your-groq-key-here"
```

**Option B — Google Gemini (free tier — https://aistudio.google.com)**

Windows PowerShell:
```powershell
$env:GEMINI_API_KEY = "AIza_your-gemini-key-here"
```
Mac/Linux:
```bash
export GEMINI_API_KEY="AIza_your-gemini-key-here"
```

**Option C — Anthropic Claude (paid — https://console.anthropic.com)**

Windows PowerShell:
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```
Mac/Linux:
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Option D — No key (MOCK mode)**

No setup needed. The assistant runs in MOCK mode automatically.
All outputs are still generated using real geological data.
Only the written interpretation text uses a fixed template instead of live AI.

---

### Step 4 — Run all notebooks in order

> ⚠️ **Must run strictly 01 → 08.** Each notebook reads files saved by the previous one.
> Set your API key **before** running notebook 08.

**Windows PowerShell:**
```powershell
python -m nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb --inplace --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/02_data_cleaning.ipynb --inplace --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/03_ml_imputation.ipynb --inplace --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/04_permeability_model.ipynb --inplace --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/05_feasibility_score.ipynb --inplace --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/06_integration.ipynb --inplace --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/07_scenario_dashboard.ipynb --inplace --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/08_ai_workflow_bonus.ipynb --inplace --ExecutePreprocessor.timeout=600
```

**Mac/Linux:**
```bash
for nb in 01_data_exploration 02_data_cleaning 03_ml_imputation 04_permeability_model \
          05_feasibility_score 06_integration 07_scenario_dashboard 08_ai_workflow_bonus; do
  jupyter nbconvert --to notebook --execute notebooks/${nb}.ipynb --inplace --ExecutePreprocessor.timeout=600
done
```

 **Success** = each notebook ends with `Writing XXXXX bytes` and no `Error` or `Traceback`.

---

## AI Backend — Notebook 08

`src/ai_workflow/assistant.py` auto-detects your environment variable and picks the backend.
You do **not** need to edit any code — just set the variable before running.

| Priority | Variable | Backend | Model | Cost |
|----------|----------|---------|-------|------|
| 1st | `GROQ_API_KEY` | Groq | llama-3.3-70b-versatile | **Free** |
| 2nd | `GEMINI_API_KEY` | Google Gemini | gemini-2.0-flash-lite | Free tier |
| 3rd | `ANTHROPIC_API_KEY` | Anthropic | claude-sonnet | Paid |
| None | — | MOCK mode | Template reports | Free |

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
│       └── assistant.py         ← GeothermalAIAssistant (Groq/Gemini/Claude/MOCK)
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

| Well   | Type     | Notes                                     |
|--------|----------|-------------------------------------------|
| BLT-01 | Vertical | Most complete logs; best ThermoGIS values |
| EVD-01 | Vertical | Sparse curves (GR, RHOB, DT only)         |
| JUT-01 | Deviated | Requires MD→TVD conversion; moderate res. |
| PKP-01 | Vertical | Moderate logs; lower perm estimates       |

---

## Key Source Files

| Module               | What it does                                        |
|----------------------|-----------------------------------------------------|
| `las_loader.py`      | Loads all 4 LAS files, strips nulls                 |
| `depth_converter.py` | Converts JUT-01 measured depth → true vertical      |
| `schema_builder.py`  | Merges all wells into unified DataFrame             |
| `imputer.py`         | Fills missing NPHI, DT, RS via XGBoost P10/P50/P90  |
| `permeability.py`    | Predicts depth-resolved perm from log data          |
| `feasibility.py`     | Scores each well 0–100 across scenarios             |
| `scenario_runner.py` | Full integration: reservoir → surface → LCOE        |
| `assistant.py`       | AI assessment reports — Groq / Gemini / Claude      |

---

## Notebook Execution Order

| Notebook | Reads | Writes |
|----------|-------|--------|
| 01 | data/raw/*.las | (in-memory EDA only) |
| 02 | raw LAS + external xlsx | data/processed/all_wells_merged.csv |
| 03 | all_wells_merged.csv | data/interim/all_wells_imputed.csv |
| 04 | all_wells_imputed.csv | data/processed/permeability_profiles.csv |
| 05 | permeability_profiles.csv + ThermoGIS | outputs/reports/feasibility_scores.csv |
| 06 | feasibility_scores.csv + LCOE.xlsx | outputs/reports/scenario_results.csv |
| 07 | scenario_results.csv | outputs/reports/scenario_dashboard.html |
| 08 | All processed data | outputs/reports/*_assessment.md/.json |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Contact

**Team Lead & AI Workflow Engineer:** Nimatallahi Masuud

**Competition:** SPE Africa Geothermal Datathon 2026

**Email:** speafricageodatathon@gmail.com
