"""
las_loader.py
=============
Loads LAS well log files using lasio, replaces null values (-999.25),
and returns clean pandas DataFrames per well.
"""

import lasio
import pandas as pd
import numpy as np
from pathlib import Path

NULL_VALUE = -999.25


def load_las(filepath: str | Path, well_name: str = None) -> pd.DataFrame:
    """
    Load a single LAS file and return a clean DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the .las file
    well_name : str, optional
        Override the well name (defaults to LAS header WELL field)

    Returns
    -------
    pd.DataFrame
        Columns: well_id, depth_m, plus all available log curves (lowercase)
        All null values (-999.25) replaced with np.nan
    """
    las = lasio.read(str(filepath))

    # Convert to DataFrame
    df = las.df().reset_index()
    df.columns = [c.strip().upper() for c in df.columns]

    # Rename depth column — lasio uses the first curve (DEPT or DEPTH or MD)
    depth_col = df.columns[0]
    df = df.rename(columns={depth_col: "DEPTH_M"})

    # Replace null sentinel values with NaN
    df = df.replace(NULL_VALUE, np.nan)

    # Add well identifier
    name = well_name or las.well.WELL.value or Path(filepath).stem
    df.insert(0, "WELL_ID", name.strip())

    # Lowercase all curve columns for consistency
    curve_cols = [c for c in df.columns if c not in ("WELL_ID", "DEPTH_M")]
    df = df.rename(columns={c: c.lower() for c in curve_cols})

    print(f"  [{name}] Loaded {len(df):,} rows | Curves: {list(df.columns[2:])}")
    return df


def load_all_wells(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load all LAS files in a directory.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing .las files

    Returns
    -------
    dict
        Keys are well names (e.g. 'BLT-01'), values are DataFrames
    """
    raw_dir = Path(raw_dir)
    las_files = sorted(raw_dir.glob("*.las"))

    if not las_files:
        raise FileNotFoundError(f"No .las files found in {raw_dir}")

    wells = {}
    print(f"Loading {len(las_files)} LAS files from {raw_dir}\n")
    for f in las_files:
        well_name = f.stem  # e.g. BLT-01
        df = load_las(f, well_name=well_name)
        wells[well_name] = df

    print(f"\nLoaded wells: {list(wells.keys())}")
    return wells


def get_curve_coverage(wells: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Returns a summary table showing % non-null values per curve per well.

    Parameters
    ----------
    wells : dict
        Output from load_all_wells()

    Returns
    -------
    pd.DataFrame
        Rows = wells, columns = log curve names, values = % non-null
    """
    records = []
    for well_name, df in wells.items():
        curve_cols = [c for c in df.columns if c not in ("WELL_ID", "DEPTH_M")]
        row = {"WELL_ID": well_name}
        for col in curve_cols:
            pct = 100.0 * df[col].notna().sum() / len(df)
            row[col] = round(pct, 1)
        records.append(row)

    coverage = pd.DataFrame(records).set_index("WELL_ID")
    return coverage
