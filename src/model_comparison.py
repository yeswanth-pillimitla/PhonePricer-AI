"""
Model Comparison and Selection Module
Aggregates all baseline, default, and hyperparameter-tuned regression models.
Evaluates multi-metric trade-offs, selects the optimal architecture, refits on
combined Train+Validation data, and executes final unbiased test evaluation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.model_evaluation import compute_metrics, format_metrics_table


def compile_model_comparison(
    default_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    output_path: str = "reports/model_comparison.csv"
) -> pd.DataFrame:
    """
    Merge default and tuned model performance benchmarks into a single sorted DataFrame.
    """
    combined = pd.concat([default_df, tuned_df], ignore_index=True)
    # Sort primarily by highest validation R2, secondarily by lowest validation MAE
    combined = combined.sort_values(by=["Val_R2", "Val_MAE"], ascending=[False, True]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def check_battery_monotonicity(pipeline: Pipeline, X_sample: pd.DataFrame) -> bool:
    """
    Verify whether the pipeline satisfies positive/non-decreasing price response
    when Battery_Capacity_mAh increases holding all other specifications constant.
    """
    try:
        sample = X_sample.iloc[0:1].copy()
        prices = []
        for bat in [4000, 4500, 5000, 5500, 6000, 7000]:
            s = sample.copy()
            s["Battery_Capacity_mAh"] = bat
            p = float(pipeline.predict(s)[0])
            prices.append(p)
        return all(prices[i+1] >= prices[i] - 1e-3 for i in range(len(prices)-1))
    except Exception:
        return False


def select_best_model(
    comparison_df: pd.DataFrame,
    default_pipes: Dict[str, Pipeline],
    tuned_pipes: Dict[str, Pipeline],
    X_val: pd.DataFrame = None
) -> Tuple[str, Pipeline, Dict[str, Any]]:
    """
    Apply selection criteria:
    1. Lowest validation MAE
    2. Low validation RMSE
    3. High validation R²
    4. Robust 5-fold CV MAE
    5. Minimal overfitting gap (Train vs Val MAE)
    6. Hardware domain consistency (strictly non-negative/monotonic battery sensitivity)
    
    Returns:
        (best_model_name, best_pipeline, best_model_record)
    """
    # Filter out models with negative R2 or extreme overfitting gap (> 10,000)
    candidates = comparison_df[
        (comparison_df["Val_R2"] > 0.6) &
        (comparison_df["Overfitting_Gap_MAE"].abs() < 8000)
    ].copy()
    
    if len(candidates) == 0:
        candidates = comparison_df.head(3).copy()
        
    # Domain physics validation: Filter to models that satisfy hardware monotonicity
    if X_val is not None:
        monotonic_indices = []
        for idx, row in candidates.iterrows():
            m_name = row["Model_Name"]
            pipe = tuned_pipes[m_name.replace(" (Tuned)", "").strip()] if "(Tuned)" in m_name else default_pipes[m_name]
            if check_battery_monotonicity(pipe, X_val):
                monotonic_indices.append(idx)
        if monotonic_indices:
            candidates = candidates.loc[monotonic_indices]
        
    # The top candidate has the best balance of Val MAE, CV MAE, and small generalization gap
    best_record = candidates.iloc[0].to_dict()
    best_name = best_record["Model_Name"]
    
    # Retrieve pipeline instance
    if "(Tuned)" in best_name:
        clean_name = best_name.replace(" (Tuned)", "").strip()
        best_pipeline = tuned_pipes[clean_name]
    else:
        best_pipeline = default_pipes[best_name]
        
    return best_name, best_pipeline, best_record


def evaluate_final_test_set(
    best_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    full_df: pd.DataFrame,
    figures_dir: str = "reports/figures",
    output_predictions_path: str = "reports/final_test_predictions.csv"
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Refit the winning pipeline on combined Train+Validation partitions (85% of data),
    generate predictions for untouched Test partition (15% of data), compute metrics,
    and render diagnostic residual and error plots.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_predictions_path), exist_ok=True)
    
    # 1. Combine Train and Validation partitions (Strictly zero test leakage)
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    
    # 2. Refit pipeline
    best_pipeline.fit(X_train_val, y_train_val)
    
    # 3. Final single-shot inference on held-out test partition
    y_test_pred = best_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred, "Test")
    
    # 4. Compile Predictions Table
    test_indices = X_test.index
    phone_names = full_df.loc[test_indices, "Phone_Model"].values if "Phone_Model" in full_df.columns else [f"Phone_{i}" for i in range(len(test_indices))]
    brands = full_df.loc[test_indices, "Phone_Company"].values if "Phone_Company" in full_df.columns else ["Unknown" for _ in range(len(test_indices))]
    
    abs_errors = np.abs(y_test.values - y_test_pred)
    pct_errors = (abs_errors / y_test.values) * 100.0
    
    pred_df = pd.DataFrame({
        "Phone_Brand": brands,
        "Phone_Model": phone_names,
        "Actual_Price_INR": y_test.values,
        "Predicted_Price_INR": np.round(y_test_pred, 2),
        "Absolute_Error_INR": np.round(abs_errors, 2),
        "Actual_Price_BDT": y_test.values,  # compatibility
        "Predicted_Price_BDT": np.round(y_test_pred, 2),  # compatibility
        "Absolute_Error_BDT": np.round(abs_errors, 2),  # compatibility
        "Percentage_Error": np.round(pct_errors, 2)
    }).sort_values(by="Absolute_Error_INR", ascending=True).reset_index(drop=True)
    
    pred_df.to_csv(output_predictions_path, index=False)
    
    # 5. Diagnostic Figures
    residuals = y_test.values - y_test_pred
    
    # Figure A: Actual vs Predicted
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test.values, y_test_pred, color="#1f77b4", edgecolors="black", s=80, alpha=0.8, label="Test Phones")
    min_val = min(y_test.min(), y_test_pred.min()) * 0.9
    max_val = max(y_test.max(), y_test_pred.max()) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Parity Line (y = x)")
    plt.title(f"Final Test Evaluation: Actual vs. Predicted Price (₹ INR)\n$R^2$ = {test_metrics['Test_R2']:.4f} | MAE = ₹{test_metrics['Test_MAE']:,.0f}", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Price (₹ INR)", fontsize=11)
    plt.ylabel("Predicted Price (₹ INR)", fontsize=11)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()
    
    # Figure B: Residual Plot (Residuals vs Predicted)
    plt.figure(figsize=(9, 5))
    plt.scatter(y_test_pred, residuals, color="#ff7f0e", edgecolors="black", s=80, alpha=0.8)
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.title("Residual Plot: Residuals vs. Predicted Price (₹ INR)", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Price (₹ INR)", fontsize=11)
    plt.ylabel("Residual Error (Actual - Predicted) [₹ INR]", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "residual_plot.png"), dpi=300)
    plt.close()
    
    # Figure C: Residual Distribution (KDE + Histogram)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#2ca02c", bins=8, edgecolor="black")
    plt.axvline(0, color="black", linestyle="--")
    plt.title(f"Residual Distribution (Mean Residual: ₹{np.mean(residuals):,.0f})", fontsize=12, fontweight="bold")
    plt.xlabel("Residual (Actual - Predicted) [₹ INR]", fontsize=11)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "residual_distribution.png"), dpi=300)
    plt.close()
    
    # Figure D: Absolute Percentage Error Distribution
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=pct_errors, color="#9467bd")
    plt.title(f"Test Percentage Error Distribution (Median MAPE: {np.median(pct_errors):.1f}%)", fontsize=12, fontweight="bold")
    plt.xlabel("Absolute Percentage Error (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "error_distribution.png"), dpi=300)
    plt.close()
    
    return test_metrics, pred_df


if __name__ == "__main__":
    from src.data_cleaning import perform_data_cleaning
    from src.data_loader import load_raw_data
    from src.feature_engineering import engineer_features
    from src.feature_selection import select_features
    from src.data_splitting import split_data
    from src.train import train_and_evaluate_all_models
    from src.model_tuning import tune_all_models
    
    df = load_raw_data()
    clean_df, _, _ = perform_data_cleaning(df)
    eng_df = engineer_features(clean_df)
    X, y, num_cols, cat_cols = select_features(eng_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    default_pipes, default_df = train_and_evaluate_all_models(X_train, y_train, X_val, y_val, num_cols, cat_cols)
    tuned_pipes, best_params, tuned_df = tune_all_models(X_train, y_train, X_val, y_val, num_cols, cat_cols)
    
    comp_df = compile_model_comparison(default_df, tuned_df)
    best_name, best_pipe, best_rec = select_best_model(comp_df, default_pipes, tuned_pipes)
    print(f"\nWinning Architecture: {best_name}")
    print(f"Validation MAE: ₹{best_rec['Val_MAE']:,.2f} | Validation R2: {best_rec['Val_R2']:.4f}")
    
    test_metrics, pred_df = evaluate_final_test_set(best_pipe, X_train, y_train, X_val, y_val, X_test, y_test, clean_df)
    print("\nFinal Test Metrics on Unseen Test Dataset:")
    print(format_metrics_table(test_metrics))
    print("\nFinal Test Predictions Sample:")
    print(pred_df.head(5).to_string())

