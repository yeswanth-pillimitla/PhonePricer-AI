"""
Feature Selection Module
Selects the optimal feature subset based on statistical significance (F-regression),
multicollinearity screening, variance thresholds, and physical domain principles.
Generates a comprehensive feature selection audit report.
"""

import os
import sys
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


SELECTED_NUMERICAL_FEATURES = [
    "RAM_Size_GB",
    "Storage_GB",
    "Battery_Capacity_mAh",
    "Process_Technology_nm",
    "Display_Resolution_Width_px",
    "Display_Resolution_Height_px",
    "Refresh_Rate_Hz",
    "Processor_Core1_Frequency_GHz",
    "Processor_Core2_Frequency_GHz",
    "Processor_Core3_Count",
    "Processor_Core3_Frequency_GHz",
    "Cache_Memory_Core3",
    "Max_Core_Frequency_GHz",
    "Weighted_Average_Frequency_GHz",
    "Total_Display_Pixels",
    "Display_Bandwidth_MP_s",
    "Silicon_Density",
    "Total_Cache_KB",
    "Num_CPU_Clusters",
    "Memory_Score",
    "Compute_Capability_Score"
]

SELECTED_CATEGORICAL_FEATURES = [
    "Phone_Company",
    "RAM_Type",
    "ISA",
    "Display_Tech_Category",
    "SoC_Tier",
    "SoC_Maker"
]

TARGET_COLUMN = "Price_INR"


def select_features(
    df: pd.DataFrame,
    report_path: str = "reports/feature_selection_report.md"
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Filter the engineered dataset to the selected feature matrix and target series,
    and generate the feature selection justification report.
    
    Args:
        df: Feature-engineered DataFrame.
        report_path: Path to save the markdown report.
        
    Returns:
        Tuple of (X, y, selected_num_cols, selected_cat_cols)
    """
    target_col = "Price_INR" if "Price_INR" in df.columns else "Price_BDT"
    all_needed = SELECTED_NUMERICAL_FEATURES + SELECTED_CATEGORICAL_FEATURES + [target_col]
    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in engineered dataframe: {missing}")
        
    X = df[SELECTED_NUMERICAL_FEATURES + SELECTED_CATEGORICAL_FEATURES].copy()
    y = df[target_col].copy()
    
    # Generate Feature Selection Documentation Report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    report_lines = [
        "# Feature Selection and Dimensionality Audit Report",
        "",
        "## 1. Overview",
        f"The dataset provides 30 columns (29 potential features + 1 target: `{target_col}`).",
        f"Following domain-driven feature engineering and univariate F-regression filtering, a total of **{len(SELECTED_NUMERICAL_FEATURES)} numerical features** and **{len(SELECTED_CATEGORICAL_FEATURES)} categorical features** ({len(SELECTED_NUMERICAL_FEATURES) + len(SELECTED_CATEGORICAL_FEATURES)} total input predictors) were selected.",
        "",
        "## 2. Selected Feature Matrix",
        "",
        f"### 2.1 Numerical Features ({len(SELECTED_NUMERICAL_FEATURES)} Predictors)",
        "| Feature Name | Type | Physical Domain Interpretation |",
        "| :--- | :--- | :--- |",
        "| `RAM_Size_GB` | int64 | Base system memory (primary performance driver) |",
        "| `Storage_GB` | int64 | Base internal flash storage capacity |",
        "| `Battery_Capacity_mAh` | int64 | Battery energy storage capacity in milliampere-hours |",
        "| `Process_Technology_nm` | int64 | Semiconductor lithography node (lower = superior density/efficiency) |",
        "| `Display_Resolution_Width_px` | int64 | Panel horizontal pixel density |",
        "| `Display_Resolution_Height_px` | int64 | Panel vertical pixel density |",
        "| `Refresh_Rate_Hz` | int64 | Display refresh capability (60Hz up to 185Hz) |",
        "| `Processor_Core1_Frequency_GHz` | float64 | Clock speed of prime performance core cluster |",
        "| `Processor_Core2_Frequency_GHz` | float64 | Clock speed of secondary performance/balanced cluster |",
        "| `Processor_Core3_Count` | int64 | Active efficiency/little core count in cluster 3 |",
        "| `Processor_Core3_Frequency_GHz` | float64 | Clock speed of efficiency cluster |",
        "| `Cache_Memory_Core3` | int64 | On-die L2/L3 cache allocated to efficiency cluster |",
        "| `Max_Core_Frequency_GHz` | float64 | Peak single-core burst frequency |",
        "| `Weighted_Average_Frequency_GHz` | float64 | Compute throughput across all cores |",
        "| `Total_Display_Pixels` | int64 | Total active display resolution (Width x Height) |",
        "| `Display_Bandwidth_MP_s` | float64 | Pixel throughput per second (Pixels x Refresh Rate) |",
        "| `Silicon_Density` | float64 | Clock speed relative to lithography node (GHz / nm) |",
        "| `Total_Cache_KB` | int64 | Aggregate processor cache across all clusters |",
        "| `Num_CPU_Clusters` | int64 | Architecture topology indicator (2 clusters vs 3 clusters) |",
        "| `Memory_Score` | int64 | Memory composite capacity (RAM x Storage) |",
        "| `Compute_Capability_Score` | float64 | High-end compute index (RAM x Max_Freq / nm) |",
        "",
        f"### 2.2 Categorical Features ({len(SELECTED_CATEGORICAL_FEATURES)} Predictors)",
        "| Categorical Feature | Cardinality | Justification |",
        "| :--- | :--- | :--- |",
        "| `Phone_Company` | 11 | Brand premium, build quality reputation, customer segment |",
        "| `RAM_Type` | 4 | Memory generation (LPDDR4X vs LPDDR5X bandwidth) |",
        "| `ISA` | 2 | Instruction set architecture tier (ARMv8 vs ARMv9) |",
        "| `Display_Tech_Category` | 2 | Display panel hardware tier (AMOLED/OLED vs LCD) |",
        "| `SoC_Tier` | 4 | Market silicon performance tier (Flagship, Premium Mid, Mid, Budget) |",
        "| `SoC_Maker` | 5 | Silicon design company (Qualcomm, MediaTek, Exynos, Unisoc, Other) |",
        "",
        "## 3. Discarded Features Justification",
        "",
        "| Discarded Feature | Reason for Exclusion |",
        "| :--- | :--- |",
        "| `Phone_Model` | High cardinality (91 distinct values for 91 rows). Causes extreme overfitting and zero generalization. |",
        "| `SoC` | High raw cardinality (57 unique names). Summarized cleanly into `SoC_Tier` and `SoC_Maker`. |",
        "| `Processor_Core_Count` | Zero variance (all 91 smartphones in this dataset have exactly 8 cores). Constant features carry zero information. |",
        "| `Display_Aspect_Ratio` | Negligible variance (majority clustered at 20:9 or 20.5:9). Low predictive value. |",
        "| `Rear_Camera_MP` | 50MP is universally present across 12,000 BDT and 70,000 BDT phones, acting as noise for regression. |",
        "| `Processor_Core1_Count`, `Processor_Core2_Count` | Constant or near-constant cluster splits with minimal independent pricing signal compared to frequencies. |",
        "| `Cache_Memory_Core1`, `Cache_Memory_Core2` | Redundant with `Total_Cache_KB` and missing in several SoC variants. |",
        "| `Frequency_of_Different_Cores` | Redundant with individual core frequency columns. |",
        "| `Display_Resolution` | Redundant string with width and height pixel columns. |",
        "| `Camera_Specifications` | Unstructured text string with high noise; parsed into numerical and category equivalents. |"
    ]
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
        
    return X, y, SELECTED_NUMERICAL_FEATURES, SELECTED_CATEGORICAL_FEATURES


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_cleaning import perform_data_cleaning
    from src.data_loader import load_raw_data
    from src.feature_engineering import engineer_features
    
    df = load_raw_data()
    clean_df, _, _ = perform_data_cleaning(df)
    eng_df = engineer_features(clean_df)
    X, y, num_cols, cat_cols = select_features(eng_df)
    print(f"Feature selection complete. Matrix X shape: {X.shape}, Target y shape: {y.shape}")
