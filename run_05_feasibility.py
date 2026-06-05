import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
from pathlib import Path

with open('config/settings.yaml') as f:
    CFG = yaml.safe_load(f)

PROC_DIR   = Path('data/processed')
EXTERNAL   = Path('data/external')
REPORT_DIR = Path('outputs/reports')
FIG_DIR    = Path('outputs/figures')
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

from src.models.feasibility import FeasibilityScorer, ScoringWeights, WellInputs

# ── 1. Load permeability profiles ──────────────────────────────────────────
print("Loading permeability_profiles.csv...")
df = pd.read_csv(PROC_DIR / 'permeability_profiles.csv')
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  is_rotliegend in columns: {'is_rotliegend' in df.columns}")

# Force boolean
df['is_rotliegend'] = df['is_rotliegend'].astype(str).str.strip().str.lower() == 'true'
print(f"  Rotliegend rows: {df['is_rotliegend'].sum()}")

# ── 2. Load ThermoGIS ───────────────────────────────────────────────────────
print("\nLoading ThermoGIS...")
thermo_raw = pd.read_excel(EXTERNAL / 'ThermoGIS Data.xlsx', sheet_name=None)
thermo = {}
for well_name, df_t in thermo_raw.items():
    df_t.columns = [str(c).strip() for c in df_t.columns]
    hr_idx = df_t[df_t.iloc[:, 0].astype(str).str.strip() == 'Property'].index
    if len(hr_idx) == 0:
        continue
    data = df_t.iloc[hr_idx[0]+1:].copy()
    data.columns = ['Property', 'Unit', 'P90', 'P50', 'P10']
    data = data.dropna(subset=['Property']).set_index('Property')
    thermo[well_name] = {
        prop: {
            'P90': float(row['P90']) if str(row['P90']) not in ('nan','') else 0.0,
            'P50': float(row['P50']) if str(row['P50']) not in ('nan','') else 0.0,
            'P10': float(row['P10']) if str(row['P10']) not in ('nan','') else 0.0,
        }
        for prop, row in data.iterrows()
    }
    t = thermo[well_name]
    print(f"  {well_name}: flow P50={t.get('Flow Rate',{}).get('P50')} m3/h, "
          f"temp P50={t.get('Temperature',{}).get('P50')}C")

# ── 3. Load target_lithologies ─────────────────────────────────────────────
target_litho = pd.read_csv(EXTERNAL / 'target_lithologies.csv')
print(f"\ntarget_lithologies: {target_litho.shape}")
dist_cols = [c for c in target_litho.columns if 'dist' in c.lower() or 'usp' in c.lower()]
print(f"  Distance columns: {dist_cols}")

# ── 4. Build perm summary from ML model ────────────────────────────────────
rotl = df[df['is_rotliegend']]
print(f"\nRotliegend rows per well:")
print(rotl.groupby('WELL_ID')['is_rotliegend'].count())

perm_summary = rotl.groupby('WELL_ID')[['k_p10_md','k_p50_md','k_p90_md']].mean().round(2)
print(f"\nML permeability (Rotliegend mean):")
print(perm_summary.to_string())

thickness = rotl.groupby('WELL_ID')['TVD_M'].agg(lambda x: x.max() - x.min()).round(0)
mean_depth = rotl.groupby('WELL_ID')['TVD_M'].mean().round(0)

# ── 5. Build WellInputs ────────────────────────────────────────────────────
print("\nBuilding WellInputs...")
inputs_list = []
for well in CFG['wells']['all']:
    if well not in thermo:
        print(f"  WARNING: {well} not in ThermoGIS")
        continue
    t = thermo[well]
    thick = float(thickness.get(well, 100))
    depth = float(mean_depth.get(well, 2000))
    dist_row = target_litho[target_litho.apply(
        lambda r: any(well.upper() in str(v).upper() for v in r.values), axis=1
    )]
    dist = float(dist_row[dist_cols[0]].iloc[0]) if (not dist_row.empty and dist_cols) else 3.0
    pk = perm_summary.loc[well] if well in perm_summary.index else None

    for scenario in ['P90', 'P50', 'P10']:
        flow = t.get('Flow Rate', {}).get(scenario, 0.0)
        temp = t.get('Temperature', {}).get(scenario, 60.0)
        if pk is not None:
            perm_map = {'P90': float(pk['k_p90_md']),
                        'P50': float(pk['k_p50_md']),
                        'P10': float(pk['k_p10_md'])}
            perm = max(perm_map[scenario], 0.001)
        else:
            perm = max(t.get('Permeability', {}).get(scenario, 1.0), 0.001)

        inputs_list.append(WellInputs(
            well_id=well, scenario=scenario,
            flow_rate_m3h=float(flow), temperature_c=float(temp),
            permeability_md=perm, thickness_m=thick,
            distance_to_usp_km=dist, depth_m=depth
        ))
        print(f"  {well} {scenario}: flow={flow} m3/h, temp={temp}C, perm={perm:.1f} mD")

print(f"\nTotal inputs built: {len(inputs_list)}")

# ── 6. Score ────────────────────────────────────────────────────────────────
w = CFG['feasibility_weights']
scorer = FeasibilityScorer(ScoringWeights(**w))
results = scorer.score_all(inputs_list)
scorer.print_summary(results)

# ── 7. Save ─────────────────────────────────────────────────────────────────
out_path = REPORT_DIR / 'feasibility_scores.csv'
results.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ── 8. Bar chart ────────────────────────────────────────────────────────────
scenario_colors = {'P10': '#2ecc71', 'P50': '#3498db', 'P90': '#e74c3c'}
wells_order = CFG['wells']['all']
x = np.arange(len(wells_order))
bar_width = 0.22

fig, ax = plt.subplots(figsize=(11, 5))
for i, scenario in enumerate(['P90', 'P50', 'P10']):
    s_data = results[results['scenario'] == scenario]
    scores = []
    for well in wells_order:
        row = s_data[s_data['well_id'] == well]
        scores.append(float(row['score_total'].iloc[0]) if not row.empty else 0.0)
    bars = ax.bar(x + i * bar_width, scores, bar_width,
                  label=scenario, color=scenario_colors[scenario], edgecolor='white')
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{score:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.axhline(50, color='orange', lw=1.5, ls='--', label='Viable (50)')
ax.axhline(70, color='green',  lw=1.5, ls='--', label='Strong (70)')
ax.set_xticks(x + bar_width)
ax.set_xticklabels(wells_order, fontsize=11, fontweight='bold')
ax.set_ylabel('Feasibility Score (0-100)')
ax.set_title('Geothermal Feasibility Scores — P90 / P50 / P10')
ax.set_ylim(0, 110)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / '05_feasibility_scores.png', dpi=150, bbox_inches='tight')
print("Saved: 05_feasibility_scores.png")

# ── 9. Heat output chart ─────────────────────────────────────────────────────
p50 = results[results['scenario'] == 'P50'].copy()
colors = ['#2ecc71' if v else '#e74c3c' for v in p50['meets_heat_target']]
fig2, ax2 = plt.subplots(figsize=(10, 5))
bars = ax2.barh(p50['well_id'], p50['heat_output_mwth'], color=colors, edgecolor='white')
ax2.axvline(CFG['targets']['heat_mwth'], color='red', lw=2, ls='--',
            label=f"Target: {CFG['targets']['heat_mwth']} MWth")
for bar, val in zip(bars, p50['heat_output_mwth']):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f} MWth', va='center', fontsize=9)
green_patch = mpatches.Patch(color='#2ecc71', label='Meets target')
red_patch   = mpatches.Patch(color='#e74c3c', label='Below target')
ax2.legend(handles=[green_patch, red_patch])
ax2.set_xlabel('Estimated Heat Output (MWth)')
ax2.set_title('Heat Output vs 10 MWth Target — P50')
ax2.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(FIG_DIR / '05_heat_output_vs_target.png', dpi=150, bbox_inches='tight')
print("Saved: 05_heat_output_vs_target.png")

print("\n=== NOTEBOOK 05 COMPLETE ===")
print(f"Results saved to: {out_path}")
print(f"Charts saved to: {FIG_DIR}")
