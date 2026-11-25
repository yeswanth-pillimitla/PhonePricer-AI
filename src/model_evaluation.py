"""
Model Evaluation Module
Computes regression evaluation metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Mean Absolute Percentage Error (MAPE)
- Median Absolute Error (MedAE)
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error
)


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    dataset_name: str = "Validation"
) -> Dict[str, float]:
    """
    Calculate all 6 required regression metrics.
    
    Args:
        y_true: Ground truth target values.
        y_pred: Model predictions.
        dataset_name: Label describing the split (e.g. 'Train', 'Validation', 'Test').
        
    Returns:
        Dict mapping metric names to computed numerical values.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_arr, y_pred_arr)
    mape = mean_absolute_percentage_error(y_true_arr, y_pred_arr) * 100.0  # as percentage
    medae = median_absolute_error(y_true_arr, y_pred_arr)
    
    prefix = f"{dataset_name}_" if dataset_name else ""
    return {
        f"{prefix}MAE": round(float(mae), 2),
        f"{prefix}MSE": round(float(mse), 2),
        f"{prefix}RMSE": round(float(rmse), 2),
        f"{prefix}R2": round(float(r2), 4),
        f"{prefix}MAPE_%": round(float(mape), 2),
        f"{prefix}MedAE": round(float(medae), 2)
    }


def format_metrics_table(metrics_dict: Dict[str, float]) -> str:
    """Format metrics dictionary into a readable markdown table."""
    lines = ["| Metric | Value | Description |", "| :--- | :--- | :--- |"]
    descriptions = {
        "MAE": "Mean Absolute Error in INR (average magnitude of errors in ₹)",
        "MSE": "Mean Squared Error (penalizes large outliers quadratically)",
        "RMSE": "Root Mean Squared Error in INR (standard deviation of residuals in ₹)",
        "R2": "R-squared Score (proportion of variance explained)",
        "MAPE_%": "Mean Absolute Percentage Error (%)",
        "MedAE": "Median Absolute Error in INR (robust to extreme errors in ₹)"
    }
    for k, v in metrics_dict.items():
        base_metric = k.split("_")[-1].replace("%", "").strip("_")
        desc = descriptions.get(base_metric, "Evaluation metric")
        if "R2" in k:
            formatted_val = f"{v:.4f}"
        elif "MAPE" in k:
            formatted_val = f"{v:.2f}%"
        else:
            formatted_val = f"₹{v:,.2f}"
        lines.append(f"| `{k}` | **{formatted_val}** | {desc} |")
    return "\n".join(lines)


if __name__ == "__main__":
    y_true = np.array([20000, 30000, 50000, 70000])
    y_pred = np.array([21000, 28000, 53000, 68000])
    res = compute_metrics(y_true, y_pred, "Test")
    print(format_metrics_table(res))

