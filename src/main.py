"""
main.py
-------
Full pipeline for NASA CMAPSS Predictive Maintenance -- RUL Prediction.

Usage:
    python src/main.py

Expects CMAPSS .txt files in ./data/ folder.
Download from: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset
from feature_engineering import (
    add_rul_labels, add_test_rul_labels,
    select_sensors, add_rolling_features,
    normalize_features, get_feature_columns,
    SELECTED_SENSORS
)
from models import train_random_forest, train_xgboost
from evaluation import evaluate_model, plot_rul_prediction, plot_feature_importance


# --- Config ------------------------------------------------------------------
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'plots')
SUBSET      = 'FD001'   # Change to FD002/FD003/FD004 as needed
WINDOW_SIZE = 5         # Rolling window for temporal features


def main():
    print("\n" + "="*60)
    print("  NASA CMAPSS Predictive Maintenance Pipeline")
    print(f"  Subset: {SUBSET}")
    print("="*60 + "\n")

    # -- 1. Load data ----------------------------------------------------------
    train_df, test_df, rul_df = load_dataset(DATA_DIR, subset=SUBSET)

    # -- 2. Feature engineering -----------------------------------------------
    train_df = add_rul_labels(train_df)
    test_df  = add_test_rul_labels(test_df, rul_df)

    train_df = select_sensors(train_df, SELECTED_SENSORS)
    test_df  = select_sensors(test_df, SELECTED_SENSORS)

    train_df = add_rolling_features(train_df, window=WINDOW_SIZE, sensors=SELECTED_SENSORS)
    test_df  = add_rolling_features(test_df, window=WINDOW_SIZE, sensors=SELECTED_SENSORS)

    train_df, test_df, scaler = normalize_features(train_df, test_df)

    # -- 3. Prepare arrays -----------------------------------------------------
    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols].values
    y_train = train_df['RUL'].values

    test_last = test_df.dropna(subset=['RUL'])
    X_test  = test_last[feature_cols].values
    y_test  = test_last['RUL'].values

    print(f"\n[Data] Training samples : {len(X_train)}")
    print(f"[Data] Test samples     : {len(X_test)}")
    print(f"[Data] Features         : {len(feature_cols)}")

    results = {}

    # -- 4. Random Forest ------------------------------------------------------
    print("\n[1/2] Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_pred  = rf_model.predict(X_test)
    rf_pred  = np.clip(rf_pred, 0, None)
    results['Random Forest'] = evaluate_model(y_test, rf_pred, 'Random Forest')

    plot_rul_prediction(y_test, rf_pred, 'Random Forest',
                        save_path=os.path.join(RESULTS_DIR, 'rf_predictions.png'))
    plot_feature_importance(rf_model, feature_cols, model_name='Random Forest',
                            save_path=os.path.join(RESULTS_DIR, 'rf_feature_importance.png'))

    # -- 5. XGBoost ------------------------------------------------------------
    print("\n[2/2] Training XGBoost...")
    try:
        xgb_model = train_xgboost(X_train, y_train)
        xgb_pred  = xgb_model.predict(X_test)
        xgb_pred  = np.clip(xgb_pred, 0, None)
        results['XGBoost'] = evaluate_model(y_test, xgb_pred, 'XGBoost')

        plot_rul_prediction(y_test, xgb_pred, 'XGBoost',
                            save_path=os.path.join(RESULTS_DIR, 'xgb_predictions.png'))
        plot_feature_importance(xgb_model, feature_cols, model_name='XGBoost',
                                save_path=os.path.join(RESULTS_DIR, 'xgb_feature_importance.png'))
    except ImportError as e:
        print(f"[WARNING] {e} -- skipping XGBoost.")

    # -- 6. Summary ------------------------------------------------------------
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Model':<20} {'RMSE':>10} {'NASA Score':>12}")
    print(f"  {'-'*44}")
    for name, metrics in results.items():
        print(f"  {name:<20} {metrics['rmse']:>10.4f} {metrics['nasa_score']:>12.2f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
