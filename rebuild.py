import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

with open('config/settings.yaml') as f:
    CFG = yaml.safe_load(f)

from src.pipeline.las_loader import load_all_wells
from src.pipeline.depth_converter import add_tvd_to_vertical_well, load_well_survey, md_to_tvd
from src.pipeline.schema_builder import build_unified_schema
from src.models.imputer import run_all_imputers
from src.models.permeability import PermeabilityModel

RAW       = Path('data/raw')
EXTERNAL  = Path('data/external')
PROC_DIR  = Path('data/processed')
INTER_DIR = Path('data/interim')
MODEL_DIR = Path('outputs/models')
for d in [PROC_DIR, INTER_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print('Step 1: Loading LAS files...')
wells = load_all_wells(RAW)
for name, df in wells.items():
    if name == 'JUT-01':
        try:
            survey = load_well_survey(EXTERNAL / 'Well Path Data.xlsx', name)
            wells[name] = md_to_tvd(df, survey)
        except Exception as e:
            print(f'  JUT-01 TVD fallback: {e}')
            wells[name] = add_tvd_to_vertical_well(df)
    else:
        wells[name] = add_tvd_to_vertical_well(df)

print('Step 2: Building schema with formation tags...')
unified = build_unified_schema(wells, EXTERNAL / 'Lithostratigraphic Data.xlsx')
rotl_count = unified['is_rotliegend'].sum()
print(f'Rotliegend rows: {rotl_count}')
unified.to_csv(PROC_DIR / 'all_wells_merged.csv', index=False)
print('Saved all_wells_merged.csv')

print('Step 3: Imputation...')
imputed = run_all_imputers(unified, save_dir=MODEL_DIR)
imputed.to_csv(INTER_DIR / 'all_wells_imputed.csv', index=False)
print('Saved all_wells_imputed.csv')

print('Step 4: Permeability profiles...')
thermo_raw = pd.read_excel(EXTERNAL / 'ThermoGIS Data.xlsx', sheet_name=None)
tg_rows = []
for wn, df_t in thermo_raw.items():
    df_t.columns = [str(c).strip() for c in df_t.columns]
    hr_idx = df_t[df_t.iloc[:,0].astype(str).str.strip() == 'Property'].index
    if len(hr_idx) == 0:
        continue
    data = df_t.iloc[hr_idx[0]+1:].copy()
    data.columns = ['Property', 'Unit', 'P90', 'P50', 'P10']
    data = data.dropna(subset=['Property']).set_index('Property')
    perm_p50 = float(data.loc['Permeability', 'P50']) if 'Permeability' in data.index else 1.0
    tg_rows.append({'WELL_ID': wn, 'PERM_P50_MD': perm_p50})
tg_df = pd.DataFrame(tg_rows)
print('ThermoGIS permeability:')
print(tg_df.to_string(index=False))

perm_model = PermeabilityModel()
perm_model.fit(imputed, tg_df)
df_perm = perm_model.predict_profiles(imputed)
df_perm.to_csv(PROC_DIR / 'permeability_profiles.csv', index=False)
perm_model.save(MODEL_DIR / 'permeability_model.pkl')

rotl_final = df_perm['is_rotliegend'].sum()
print(f'Rotliegend rows in final file: {rotl_final}')
if rotl_final > 0:
    print('ALL DONE - now run notebook 05')
else:
    print('ERROR - is_rotliegend still 0, check schema_builder.py')
