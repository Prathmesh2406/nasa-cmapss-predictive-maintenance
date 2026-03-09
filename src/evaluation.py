"""
evaluation.py
-------------
Evaluation metrics for RUL prediction:
  - RMSE (Root Mean Squared Error)
  - NASA PHM Score Function (asymmetric penalty)
  - Prediction visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the NASA PHM scoring function.

    The NASA score penalizes late predictions (positive error) more heavily
    than early predictions (negative error), reflecting the real-world cost
    of missing an impending failure.

    Score = sum(exp(d/13) - 1)  for d >= 0  (late prediction)
            sum(exp(-d/10) - 1) for d < 0   (early prediction)

    where d = y_pred - y_true

    Lower score is better. Perfect prediction -> score = 0.

    Args:
        y_true: Ground truth RUL values
        y_pred: Predicted RUL values

    Returns:
        NASA score (float)
    """
    d = y_pred - y_true
    score = np.where(d >= 0,
                     np.exp(d / 13) - 1,
                     np.exp(-d / 10) - 1)
    return float(np.sum(score))


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   model_name: str = 'Model') -> dict:
    r = rmse(y_true, y_pred)
    s = nasa_score(y_true, y_pred)
    print(f"\n{'='*40}")
    print(f"  {model_name} Results")
    print(f"{'='*40}")
    print(f"  RMSE        : {r:.4f}")
    print(f"  NASA Score  : {s:.2f}")
    print(f"{'='*40}\n")
    return {'rmse': r, 'nasa_score': s}


def plot_rul_prediction(y_true, y_pred, model_name='Model', save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} -- RUL Prediction vs Actual', fontsize=14, fontweight='bold')
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, color='steelblue', s=30, edgecolors='none')
    max_val = max(y_true.max(), y_pred.max()) + 5
    ax.plot([0, max_val], [0, max_val], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual RUL (cycles)')
    ax.set_ylabel('Predicted RUL (cycles)')
    ax.set_title('Predicted vs Actual RUL')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    errors = y_pred - y_true
    ax.hist(errors, bins=30, color='coral', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='black', linestyle='--', lw=1.5)
    ax.axvline(np.mean(errors), color='red', linestyle='-', lw=1.5,
               label=f'Mean error: {np.mean(errors):.1f}')
    ax.set_xlabel('Prediction Error (cycles)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sensor_degradation(df, unit_id=1, sensors=None, save_path=None):
    if sensors is None:
        sensors = [c for c in df.columns if c.startswith('sensor_')][:6]
    unit_data = df[df['unit_id'] == unit_id].sort_values('cycle')
    n = len(sensors)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    fig.suptitle(f'Engine #{unit_id} -- Sensor Degradation Over Lifetime', fontsize=13, fontweight='bold')
    axes = axes.flatten()
    for i, sensor in enumerate(sensors):
        ax = axes[i]
        ax.plot(unit_data['cycle'], unit_data[sensor], color='steelblue', lw=1.5, alpha=0.8)
        ax.set_title(sensor.replace('_', ' ').title())
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15, model_name='Model', save_path=None):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices][::-1], color='steelblue', edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{model_name} -- Top {top_n} Feature Importances')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
