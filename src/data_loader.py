"""
data_loader.py
--------------
Load and parse NASA CMAPSS turbofan engine degradation dataset.
"""

import pandas as pd
import numpy as np
import os


# Column names for CMAPSS dataset
COLUMN_NAMES = (
    ['unit_id', 'cycle'] +
    [f'setting_{i}' for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)


def load_dataset(data_dir: str, subset: str = 'FD001'):
    """
    Load train, test, and RUL files for a given CMAPSS subset.

    Args:
        data_dir: Path to folder containing CMAPSS .txt files
        subset: One of 'FD001', 'FD002', 'FD003', 'FD004'

    Returns:
        train_df, test_df, rul_df (DataFrames)
    """
    train_path = os.path.join(data_dir, f'train_{subset}.txt')
    test_path  = os.path.join(data_dir, f'test_{subset}.txt')
    rul_path   = os.path.join(data_dir, f'RUL_{subset}.txt')

    train_df = _read_cmapss_file(train_path)
    test_df  = _read_cmapss_file(test_path)
    rul_df   = pd.read_csv(rul_path, header=None, names=['RUL'])

    print(f"[INFO] Loaded {subset}:")
    print(f"  Train: {train_df['unit_id'].nunique()} engines, {len(train_df)} cycles")
    print(f"  Test:  {test_df['unit_id'].nunique()} engines, {len(test_df)} cycles")

    return train_df, test_df, rul_df


def _read_cmapss_file(path: str) -> pd.DataFrame:
    """Parse a raw CMAPSS space-delimited .txt file into a DataFrame."""
    df = pd.read_csv(
        path,
        sep=r'\s+',
        header=None,
        names=COLUMN_NAMES,
        engine='python'
    )
    # Drop trailing NaN columns if any
    df = df.dropna(axis=1, how='all')
    return df


def get_sensor_columns() -> list:
    """Return list of all 21 sensor column names."""
    return [f'sensor_{i}' for i in range(1, 22)]


def get_setting_columns() -> list:
    """Return list of 3 operational setting column names."""
    return [f'setting_{i}' for i in range(1, 4)]
