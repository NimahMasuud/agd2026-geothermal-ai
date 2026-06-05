"""
depth_converter.py
==================
Converts Measured Depth (MD) to True Vertical Depth (TVD) for deviated wells.
JUT-01 is a deviated well — all depth-based calculations must use TVD.

Method: Minimum Curvature (industry standard for directional wells)
The Well Path Data.xlsx already contains a pre-computed TVD column; this
module interpolates from that survey table and merges TVD into LAS DataFrames.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_well_survey(survey_path: str | Path, well_name: str) -> pd.DataFrame:
    """
    Load directional survey for a specific well from Well Path Data.xlsx.

    Parameters
    ----------
    survey_path : str or Path
        Path to Well Path Data.xlsx
    well_name : str
        Well name to filter (e.g. 'JUT-01')

    Returns
    -------
    pd.DataFrame
        Columns: MD, INC, AZI, TVD, X_OFFSET, Y_OFFSET
    """
    xl = pd.ExcelFile(str(survey_path))
    print(f"Available sheets in Well Path Data: {xl.sheet_names}")

    # Try to find the sheet matching the well name
    target_sheet = None
    for sheet in xl.sheet_names:
        if well_name.upper() in sheet.upper() or sheet.upper() in well_name.upper():
            target_sheet = sheet
            break

    if target_sheet is None:
        # Load first sheet if no match — inspect manually
        target_sheet = xl.sheet_names[0]
        print(f"  Warning: No sheet matching '{well_name}' — loading '{target_sheet}'")

    df = pd.read_excel(survey_path, sheet_name=target_sheet, header=0)
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
    print(f"  Survey columns for {well_name}: {list(df.columns)}")
    return df


def md_to_tvd(las_df: pd.DataFrame, survey_df: pd.DataFrame,
              md_col: str = "DEPTH_M") -> pd.DataFrame:
    """
    Interpolate TVD from a directional survey table and add it to a LAS DataFrame.

    Uses linear interpolation between survey stations (sufficient accuracy
    when station spacing is fine relative to log sample interval).

    Parameters
    ----------
    las_df : pd.DataFrame
        Output from load_las() — must contain DEPTH_M (measured depth)
    survey_df : pd.DataFrame
        Output from load_well_survey() — must contain MD and TVD columns
    md_col : str
        Name of the MD column in las_df (default 'DEPTH_M')

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'TVD_M' column added
    """
    # Identify MD and TVD columns in survey (flexible naming)
    survey_cols = survey_df.columns.tolist()

    md_survey_col = next(
        (c for c in survey_cols if "MD" in c or "DEPTH" in c or "MEAS" in c), None
    )
    tvd_survey_col = next(
        (c for c in survey_cols if "TVD" in c or "TRUE" in c or "VERT" in c), None
    )

    if md_survey_col is None or tvd_survey_col is None:
        raise ValueError(
            f"Cannot find MD/TVD columns in survey. Available: {survey_cols}"
        )

    print(f"  Using survey columns: MD='{md_survey_col}', TVD='{tvd_survey_col}'")

    survey_clean = survey_df[[md_survey_col, tvd_survey_col]].dropna()
    survey_clean = survey_clean.sort_values(md_survey_col)

    # Interpolate TVD at each LAS depth point
    tvd_interp = np.interp(
        las_df[md_col].values,
        survey_clean[md_survey_col].values,
        survey_clean[tvd_survey_col].values,
        left=np.nan,
        right=np.nan
    )

    df_out = las_df.copy()
    df_out.insert(2, "TVD_M", tvd_interp)
    print(f"  TVD interpolated: {df_out['TVD_M'].notna().sum():,} valid values")
    return df_out


def add_tvd_to_vertical_well(las_df: pd.DataFrame) -> pd.DataFrame:
    """
    For vertical wells (BLT-01, EVD-01, PKP-01), TVD ≈ MD.
    Adds a TVD_M column equal to DEPTH_M.

    Parameters
    ----------
    las_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df_out = las_df.copy()
    df_out.insert(2, "TVD_M", df_out["DEPTH_M"])
    return df_out
