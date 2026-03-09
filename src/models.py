"""
models.py
---------
Machine learning models for RUL prediction:
  - Random Forest Regressor
  - XGBoost Gradient Boosting
  - LSTM Neural Network (time-series)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                         n_estimators: int = 200,
                         max_depth: int = 15,
                         random_state: int = 42):
    """Train a Random Forest regressor for RUL prediction."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"[RF] Trained with {n_estimators} estimators, max_depth={max_depth}")
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   n_estimators: int = 300,
                   learning_rate: float = 0.05,
                   max_depth: int = 6,
                   random_state: int = 42):
    """Train an XGBoost gradient boosting regressor for RUL prediction."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    print(f"[XGB] Trained with {n_estimators} rounds, lr={learning_rate}, depth={max_depth}")
    return model


def prepare_lstm_sequences(df: pd.DataFrame, feature_cols: list,
                             sequence_length: int = 30):
    """Convert tabular data to overlapping sequences for LSTM input."""
    X_list, y_list = [], []

    for unit_id, group in df.groupby('unit_id'):
        group = group.sort_values('cycle')
        features = group[feature_cols].values
        labels   = group['RUL'].values

        for i in range(len(features) - sequence_length + 1):
            X_list.append(features[i: i + sequence_length])
            y_list.append(labels[i + sequence_length - 1])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_lstm_model(sequence_length: int, n_features: int,
                      units: int = 64, dropout_rate: float = 0.2):
    """Build a stacked LSTM model for RUL regression."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        raise ImportError("tensorflow not installed. Run: pip install tensorflow")

    model = Sequential([
        LSTM(units, input_shape=(sequence_length, n_features), return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units // 2),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    print(f"[LSTM] Model built: seq_len={sequence_length}, features={n_features}, units={units}")
    return model


def train_lstm(model, X_train: np.ndarray, y_train: np.ndarray,
               epochs: int = 50, batch_size: int = 256,
               validation_split: float = 0.1):
    """Train the LSTM model."""
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        raise ImportError("tensorflow not installed.")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    print(f"[LSTM] Training complete.")
    return model, history
