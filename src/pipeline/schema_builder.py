"""
schema_builder.py
=================
Builds a unified, standardised DataFrame from all 4 wells.
Also tags each row with its formation (from the Lithostratigraphic Data.xlsx)
and flags which rows belong to the Rotliegend target interval.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# --- Canonical column schema ---
# All wells are mapped to these standard names where available.
# Curves not present for a well will remain as NaN.
SCHEMA = {
    # Standard name  : possible LAS mnemonics (in priority order)
    "gr_api":         ["GR", "GRC", "GRD"],
    "rhob_gcc":       ["RHOB", "RHOZ", "DEN"],
    "drho_gcc":       ["DRHO", "DRHB"],
    "nphi_vv":        ["NPHI", "PHIS", "PHIT"],
    "dt_usft":        ["DT", "DTC", "AC"],
    "dts_usft":       ["DTS", "ACS"],
    "rd_ohmm":        ["RD", "ILD", "RT", "RILD"],
    "rm_ohmm":        ["RM", "ILM", "RILM"],
    "rs_ohmm":        ["RS", "SFL", "RSFL"],
}


def _map_curves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw LAS curve names to canonical schema names.
    Returns a DataFrame with canonical column names only.
    """
    available = {c.upper(): c for c in df.columns}
    mapped = {}

    for canonical, aliases in SCHEMA.items():
        found = None
        for alias in aliases:
            if alias.upper() in available:
                found = available[alias.upper()]
                break
        if found:
            mapped[canonical] = df[found]
        else:
            mapped[canonical] = np.nan  # column will be all NaN

    result = pd.DataFrame(mapped, index=df.index)
    return result


def load_formation_tops(litho_path: str | Path) -> pd.DataFrame:
    """
    Load formation top/bottom depths from Lithostratigraphic Data.xlsx.
    Each sheet is named after a well (e.g. BLT-01, EVD-01).
    Columns: Stratigrafical unit, Top (m), Bottom (m)

    Returns
    -------
    pd.DataFrame
        Columns: well_id, formation, top_m, bottom_m
    """
    xl = pd.ExcelFile(str(litho_path))
    print(f"Lithostratigraphic sheets: {xl.sheet_names}")

    frames = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(litho_path, sheet_name=sheet, header=0)
        df.columns = [str(c).strip() for c in df.columns]

        # Find formation name, top, bottom columns by content
        form_col = next((c for c in df.columns
                         if "strat" in c.lower() or "unit" in c.lower()
                         or "formation" in c.lower() or "member" in c.lower()), None)
        top_col  = next((c for c in df.columns if "top" in c.lower()), None)
        bot_col  = next((c for c in df.columns
                         if "bottom" in c.lower() or "base" in c.lower()), None)

        if form_col and top_col and bot_col:
            clean = pd.DataFrame({
                "well_id":   sheet,          # sheet name IS the well name
                "formation": df[form_col].astype(str).str.strip(),
                "top_m":     pd.to_numeric(df[top_col],  errors="coerce"),
                "bottom_m":  pd.to_numeric(df[bot_col], errors="coerce"),
            }).dropna(subset=["top_m", "bottom_m"])
            frames.append(clean)
            print(f"  {sheet}: {len(clean)} formation intervals loaded")
        else:
            print(f"  WARNING: Could not parse sheet {sheet} — cols: {list(df.columns)}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"Total formation intervals: {len(combined)}")
    return combined


def tag_formations(las_df: pd.DataFrame, tops_df: pd.DataFrame,
                   well_id_col: str = "well_id") -> pd.DataFrame:
    """
    Add a 'formation' column to a LAS DataFrame based on TVD_M depth.
    Rows between top_m and bottom_m get the formation name.
    Rows outside all intervals get 'UNKNOWN'.

    Parameters
    ----------
    las_df : pd.DataFrame
        Must contain WELL_ID, TVD_M
    tops_df : pd.DataFrame
        Output from load_formation_tops()

    Returns
    -------
    pd.DataFrame
        Input with 'formation' and 'is_rotliegend' columns added
    """
    df = las_df.copy()
    df["formation"] = "UNKNOWN"

    well = df["WELL_ID"].iloc[0]

    # Filter tops for this well — try flexible matching
    # New format: well_id, formation, top_m, bottom_m
    if "well_id" in tops_df.columns:
        well_tops = tops_df[tops_df["well_id"].str.upper() == well.upper()].copy()
        top_col, bot_col, form_col = "top_m", "bottom_m", "formation"
    else:
        well_tops = tops_df[
            tops_df.apply(lambda r: any(
                well.upper() in str(v).upper() for v in r.values
            ), axis=1)
        ]
        cols = well_tops.columns.tolist()
        top_col  = next((c for c in cols if "top"  in c.lower()), None)
        bot_col  = next((c for c in cols if "bot"  in c.lower() or "base" in c.lower()), None)
        form_col = next((c for c in cols if "form" in c.lower() or "unit" in c.lower()), None)

    if well_tops.empty:
        print(f"  Warning: No formation tops found for well '{well}'")
        df["is_rotliegend"] = False
        return df

    if not all([top_col, bot_col, form_col]):
        print(f"  Warning: Could not identify top/bottom/formation cols")
        df["is_rotliegend"] = False
        return df

    depth_col = "TVD_M" if "TVD_M" in df.columns else "DEPTH_M"

    for _, row in well_tops.iterrows():
        try:
            top  = float(row[top_col])
            bot  = float(row[bot_col])
            name = str(row[form_col]).strip()
            mask = (df[depth_col] >= top) & (df[depth_col] <= bot)
            df.loc[mask, "formation"] = name
        except (ValueError, TypeError):
            continue

    df["is_rotliegend"] = df["formation"].str.upper().str.contains(
        "ROTLIEG|ROTL|SLOCHTEREN|HELLEVOETSLUIS|MAURITS", na=False
    )

    rotl_count = df["is_rotliegend"].sum()
    print(f"  [{well}] Rotliegend interval: {rotl_count:,} rows tagged")
    return df


def build_unified_schema(wells: dict[str, pd.DataFrame],
                         litho_path: str | Path) -> pd.DataFrame:
    """
    Main entry point: combine all wells into a single, standardised DataFrame.

    Parameters
    ----------
    wells : dict
        Output from load_all_wells() — already has TVD_M added
    litho_path : str or Path
        Path to Lithostratigraphic Data.xlsx

    Returns
    -------
    pd.DataFrame
        Unified DataFrame with canonical curves, formation tags, and Rotliegend flag
    """
    tops_df = load_formation_tops(litho_path)
    frames = []

    for well_name, df in wells.items():
        print(f"\nBuilding schema for {well_name}...")

        # Base columns
        base = df[["WELL_ID", "DEPTH_M", "TVD_M"]].copy()

        # Map curves to canonical names
        curves = _map_curves(df)

        # Combine
        combined = pd.concat([base.reset_index(drop=True),
                               curves.reset_index(drop=True)], axis=1)

        # Tag formations
        combined = tag_formations(combined, tops_df)

        frames.append(combined)

    unified = pd.concat(frames, ignore_index=True)

    print(f"\nUnified schema: {len(unified):,} rows x {len(unified.columns)} cols")
    print(f"Wells: {unified['WELL_ID'].unique().tolist()}")
    print(f"Rotliegend rows: {unified['is_rotliegend'].sum():,}")
    return unified
