"""
Data Loader Module
Loads raw dataset and configuration settings safely.
"""

import os
from typing import Dict, Any, Tuple
import yaml
import pandas as pd


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(data_path: str = "data/raw/coa-12.csv") -> pd.DataFrame:
    """
    Load raw smartphone CSV dataset and perform initial schema checks.
    
    Args:
        data_path: Path to the raw CSV file.
        
    Returns:
        pd.DataFrame containing the raw smartphone records.
    """
    if not os.path.exists(data_path):
        if os.path.exists("Coa-12.csv"):
            data_path = "Coa-12.csv"
        elif os.path.exists("coa-12.csv"):
            data_path = "coa-12.csv"
        elif os.path.exists("data/raw/coa-12.csv"):
            data_path = "data/raw/coa-12.csv"
        else:
            raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    df = pd.read_csv(data_path)
    
    target_col = "Price_INR" if "Price_INR" in df.columns else "Price_BDT"
    
    expected_cols = [
        "Phone_Company", "Phone_Model", "SoC", "Processor_Core_Count",
        "Processor_Core1", "Processor_Core1_Count", "Processor_Core1_Frequency_GHz",
        "Processor_Core2", "Processor_Core2_Count", "Processor_Core2_Frequency_GHz",
        "Processor_Core3", "Processor_Core3_Count", "Processor_Core3_Frequency_GHz",
        "RAM_Type", "RAM_Size_GB",
        "Cache_Memory_Core1", "Cache_Memory_Core2", "Cache_Memory_Core3",
        "Storage_GB", "Battery_Capacity_mAh",
        "ISA", "Frequency_of_Different_Cores", "Process_Technology_nm",
        "Display_Type", "Display_Resolution", "Display_Resolution_Width_px",
        "Display_Resolution_Height_px", "Refresh_Rate_Hz", "Camera_Specifications",
        target_col
    ]
    
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Data successfully loaded. Shape: {df.shape}")

