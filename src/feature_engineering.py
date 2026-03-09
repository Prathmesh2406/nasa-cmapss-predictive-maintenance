"""
feature_engineering.py
-----------------------
RUL label generation, sensor selection, rolling statistics, and normalization
for the NASA CMAPSS predictive maintenance dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Sensors with near-zero variance (constant across all cycles) -- drop these
# Identified empirically from FD001; validated across subsets
LOW_INFO_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                    'sensor_16', 'sensor_18', 'sensor_19']

# Sensors known to carry degradation signal
SELECTED_SENSORS = [f'sensor_{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]

# Piecewise linear RUL cap -- engines don't degrade meaningfully in early life
RUL_CLIP_MAX = 125


def add_rul_labels(df: pd.DataFrame, clip_max: int = RUL_CLIP_MAX) -> pd.DataFrame:
    """
    Generate Remaining Useful Life (RUL) labels for training data.

    Uses a piecewise linear degradation model:
    - RUL is capped at clip_max for early cycles (flat region)
    - Decreases linearly after that

    Args:
        df: Training DataFrame with 'unit_id' and 'cycle' columns
        clip_max: Maximum RUL value (cap for early healthy cycles)

    Returns:
        DataFrame with added 'RUL' column
    """
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']

    df = df.merge(max_cycles, on='unit_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=clip_max)
    df.drop(columns=['max_cycle'], inplace=True)
    return df


def add_test_rul_labels(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach ground-truth RUL values to test data (last cycle of each engine).
    """
    last_cycles = test_df.groupby('unit_id')['cycle'].max().reset_index()
    last_cycles.columns = ['unit_id', 'max_cycle']

    rul_df = rul_df.copy()
    rul_df['unit_id'] = range(1, len(rul_df) + 1)

    test_df = test_df.merge(last_cycles, on='unit_id', how='left')
    test_df = test_df.merge(rul_df, on='unit_id', how='left')

    test_df['RUL'] = np.where(
        test_df['cycle'] == test_df['max_cycle'],
        test_df['RUL'],
        np.nan
    )
    test_df.drop(columns=['max_cycle'], inplace=True)
    return test_df


def select_sensors(df: pd.DataFrame, sensors: list = SELECTED_SENSORS) -> pd.DataFrame:
    """Keep only informative sensors and drop low-variance ones."""
    keep_cols = ['unit_id', 'cycle'] + \
                [c for c in df.columns if c.startswith('setting_')] + \
                [s for s in sensors if s in df.columns]

    if 'RUL' in df.columns:
        keep_cols.append('RUL')

    return df[keep_cols]


def add_rolling_features(df: pd.DataFrame, window: int = 5,
                          sensors: list = SELECTED_SENSORS) -> pd.DataFrame:
    """Add rolling mean and std features over the last window cycles per engine."""
    df = df.sort_values(['unit_id', 'cycle']).copy()

    for sensor in sensors:
        if sensor not in df.columns:
            continue
        grouped = df.groupby('unit_id')[sensor]
        df[f'{sensor}_roll_mean'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'{sensor}_roll_std'] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )

    return df


def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        exclude_cols: list = None):
    """Apply Min-Max normalization. Scaler fit on training data only."""
    if exclude_cols is None:
        exclude_cols = ['unit_id', 'cycle', 'RUL']

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    scaler = MinMaxScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    return train_df, test_df, scaler


def get_feature_columns(df: pd.DataFrame, exclude_cols: list = None) -> list:
    """Return list of feature columns (excluding metadata and target)."""
    if exclude_cols is None:
        exclude_cols = ['unit_id', 'cycle', 'RUL']
    return [c for c in df.columns if c not in exclude_cols]
