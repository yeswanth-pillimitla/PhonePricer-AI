"""
Data Splitting Module
Implements reproducible, quantile-stratified train/validation/test splits (70/15/15)
to preserve price tier representation, with zero leakage across partitions.
"""

import os
import sys
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: int = 14,
    test_size: int = 14,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split feature matrix and target series into three strictly isolated subsets:
    - Training set: 70% (63 records)
    - Validation set: 15% (14 records)
    - Test set: 15% (14 records)
    
    Uses quantile-binned stratification on price to ensure balanced representation
    of budget, mid-range, and flagship price tiers across all partitions.
    
    Args:
        X: Feature matrix.
        y: Target series (Price_BDT).
        val_size: Number of validation samples.
        test_size: Number of test samples.
        random_state: Random seed for exact reproducibility.
        
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Create 5 quantile bins for stratification
    price_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    
    # First split: Hold out test set (14 samples)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=price_bins
    )
    
    # Second split: Separate train and validation from train_val (14 validation samples)
    bins_train_val = price_bins.loc[X_train_val.index]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=bins_train_val
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_split_diagnostics(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series
) -> None:
    """Print summary statistics for the dataset splits."""
    print("=" * 60)
    print("DATASET SPLIT DIAGNOSTICS (70% Train / 15% Val / 15% Test)")
    print("=" * 60)
    print(f"Training set:   {len(y_train)} samples | Mean Price: ₹{y_train.mean():,.0f} | Std: ₹{y_train.std():,.0f}")
    print(f"Validation set: {len(y_val)} samples | Mean Price: ₹{y_val.mean():,.0f} | Std: ₹{y_val.std():,.0f}")
    print(f"Test set:       {len(y_test)} samples | Mean Price: ₹{y_test.mean():,.0f} | Std: ₹{y_test.std():,.0f}")
    print("=" * 60)


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_cleaning import perform_data_cleaning
    from src.data_loader import load_raw_data
    from src.feature_engineering import engineer_features
    from src.feature_selection import select_features
    
    df = load_raw_data()
    clean_df, _, _ = perform_data_cleaning(df)
    eng_df = engineer_features(clean_df)
    X, y, _, _ = select_features(eng_df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print_split_diagnostics(y_train, y_val, y_test)

