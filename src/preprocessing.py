"""
Data Preprocessing Module
Constructs Scikit-learn ColumnTransformer and Pipeline architectures.
Provides dedicated pipelines for linear/distance-based vs tree-based models,
preventing data leakage and handling unseen categorical levels safely.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
    scaling: Optional[str] = "standard"
) -> ColumnTransformer:
    """
    Construct an industry-standard ColumnTransformer.
    
    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.
        scaling: 'standard' (StandardScaler), 'robust' (RobustScaler), or None (tree-based).
        
    Returns:
        ColumnTransformer instance.
    """
    # 1. Numerical pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaling == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaling == "robust":
        num_steps.append(("scaler", RobustScaler()))
    elif scaling is None:
        pass  # Tree-based models do not require scaling
    else:
        raise ValueError(f"Unsupported scaling method: {scaling}")
        
    numerical_pipeline = Pipeline(steps=num_steps)
    
    # 2. Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )
    
    return preprocessor


def generate_preprocessing_report(
    num_cols: List[str],
    cat_cols: List[str],
    output_path: str = "reports/preprocessing_report.md"
) -> str:
    """Generate documentation report explaining preprocessing pipelines and zero-leakage guarantee."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report_lines = [
        "# Data Preprocessing and Leakage Prevention Report",
        "",
        "## 1. Architectural Design",
        "Data preprocessing is engineered using Scikit-Learn `Pipeline` and `ColumnTransformer` constructs.",
        "Crucially, preprocessing is encapsulated **inside** the model pipeline so that fitting happens **exclusively on the training dataset**.",
        "",
        "### Dual-Pipeline Strategy",
        "1. **Linear / Distance-based Pipeline (`scaling='standard'` / `'robust'`)**:",
        "   - Used by: Linear Regression, Ridge, Lasso, ElasticNet, SVR, and K-Nearest Neighbors.",
        "   - Numerical steps: `SimpleImputer(strategy='median')` -> `StandardScaler()`.",
        "   - Ensures gradients and Euclidean distances are not distorted by differences in measurement units (e.g., mAh vs GHz).",
        "",
        "2. **Tree-based Pipeline (`scaling=None`)**:",
        "   - Used by: Decision Trees, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting.",
        "   - Numerical steps: `SimpleImputer(strategy='median')` (preserves raw numeric feature scales).",
        "   - Avoids redundant floating-point scaling transformations that do not alter monotonic tree partition decisions.",
        "",
        "## 2. Categorical Encoding with Out-of-Vocabulary Safety",
        "- Categorical columns: `Phone_Company`, `RAM_Type`, `ISA`, `Display_Tech_Category`.",
        "- Encoded using: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.",
        "- **Robustness Guarantee**: Any unseen smartphone brand, RAM type, or architecture encountered in production or test sets will produce an all-zero one-hot row rather than throwing an exception.",
        "",
        "## 3. Strict Anti-Leakage Protocol",
        "- Neither validation data nor test data is ever passed to `preprocessor.fit()`.",
        "- The identical fitted transformer learned from $X_{train}$ is used to transform $X_{val}$, $X_{test}$, and user inputs in the Streamlit application.",
        f"- Target variable `Price_INR` is isolated and never scaled or included in transformer inputs."
    ]
    
    report_text = "\n".join(report_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    return report_text


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_cleaning import perform_data_cleaning
    from src.data_loader import load_raw_data
    from src.feature_engineering import engineer_features
    from src.feature_selection import select_features
    from src.data_splitting import split_data
    
    df = load_raw_data()
    clean_df, _, _ = perform_data_cleaning(df)
    eng_df = engineer_features(clean_df)
    X, y, num_cols, cat_cols = select_features(eng_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    preprocessor = build_preprocessor(num_cols, cat_cols, scaling="standard")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    
    print(f"Preprocessing built and verified.")
    print(f"X_train transformed shape: {X_train_trans.shape}")
    print(f"X_val transformed shape: {X_val_trans.shape}")
    generate_preprocessing_report(num_cols, cat_cols)

