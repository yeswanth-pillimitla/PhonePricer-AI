"""
Feature Engineering Module
Derives robust, physically meaningful smartphone engineering features:
- Core frequency metrics (Maximum and Weighted Average)
- Total display pixels, aspect ratio, and display bandwidth
- Silicon density (lithography efficiency) and total cache memory
- Normalized display technology category (AMOLED/OLED vs LCD)
- Primary rear camera megapixels extraction
- CPU cluster count architecture indicator
- Memory capacity score and compute capability score
- System-on-Chip (SoC) tier and manufacturer categorization
"""

import os
import sys
import re
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def extract_rear_camera_mp(spec: str) -> float:
    """Extract primary rear camera megapixels using regex."""
    if pd.isna(spec):
        return 50.0  # Median default if missing
    match = re.search(r"(\d+(?:\.\d+)?)\s*MP", str(spec), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 50.0


def categorize_display_type(display_str: str) -> str:
    """Normalize raw display text into standard technology category."""
    if pd.isna(display_str):
        return "LCD"
    text = str(display_str).upper()
    if "AMOLED" in text or "OLED" in text:
        return "AMOLED_OLED"
    elif "LCD" in text or "PLS" in text or "IPS" in text:
        return "LCD"
    return "LCD"


def categorize_soc(soc_str: str) -> Tuple[str, str]:
    """
    Derive SoC performance tier and semiconductor manufacturer from SoC name string.
    
    Tiers:
    - Flagship: Snapdragon 8 series, Dimensity 9000 series, Snapdragon Elite
    - Premium_Mid: Snapdragon 7 series, Dimensity 8000/7000 series, Exynos 1380
    - Mid: Snapdragon 6/4 series, Dimensity 6000 series, Helio G90/G100 series
    - Budget: Helio G30 series, Unisoc, entry-level silicon
    """
    if pd.isna(soc_str):
        return "Mid", "Other"
        
    s_low = str(soc_str).lower()
    
    # Tier classification
    if "snapdragon 8" in s_low or "dimensity 9" in s_low or "elite" in s_low:
        tier = "Flagship"
    elif "snapdragon 7" in s_low or "dimensity 8" in s_low or "dimensity 7" in s_low or "exynos 1380" in s_low:
        tier = "Premium_Mid"
    elif "snapdragon 6" in s_low or "snapdragon 4" in s_low or "dimensity 6" in s_low or "helio g9" in s_low or "helio g1" in s_low:
        tier = "Mid"
    else:
        tier = "Budget"
        
    # Manufacturer classification
    if "snapdragon" in s_low or "qualcomm" in s_low:
        maker = "Qualcomm"
    elif "dimensity" in s_low or "helio" in s_low or "mediatek" in s_low:
        maker = "MediaTek"
    elif "exynos" in s_low:
        maker = "Exynos"
    elif "unisoc" in s_low:
        maker = "Unisoc"
    else:
        maker = "Other"
        
    return tier, maker


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply engineering transformations to derive informative continuous and categorical features.
    
    Args:
        df: Input cleaned DataFrame.
        
    Returns:
        DataFrame with added engineered feature columns.
    """
    df_eng = df.copy()
    
    # 1. Maximum Processor Core Frequency (GHz)
    df_eng["Max_Core_Frequency_GHz"] = df_eng[[
        "Processor_Core1_Frequency_GHz",
        "Processor_Core2_Frequency_GHz",
        "Processor_Core3_Frequency_GHz"
    ]].max(axis=1)
    
    # 2. Weighted Average Processor Core Frequency (GHz)
    total_cores = (
        df_eng["Processor_Core1_Count"] +
        df_eng["Processor_Core2_Count"] +
        df_eng["Processor_Core3_Count"]
    ).replace(0, 8)
    
    weighted_freq_sum = (
        df_eng["Processor_Core1_Count"] * df_eng["Processor_Core1_Frequency_GHz"] +
        df_eng["Processor_Core2_Count"] * df_eng["Processor_Core2_Frequency_GHz"] +
        df_eng["Processor_Core3_Count"] * df_eng["Processor_Core3_Frequency_GHz"]
    )
    df_eng["Weighted_Average_Frequency_GHz"] = (weighted_freq_sum / total_cores).round(3)
    
    # 3. Total Display Pixels (Width x Height)
    df_eng["Total_Display_Pixels"] = (
        df_eng["Display_Resolution_Width_px"] * df_eng["Display_Resolution_Height_px"]
    )
    
    # 4. Display Aspect Ratio (Height / Width)
    df_eng["Display_Aspect_Ratio"] = (
        df_eng["Display_Resolution_Height_px"] / df_eng["Display_Resolution_Width_px"].replace(0, 1080)
    ).round(2)
    
    # 5. Display Bandwidth (MegaPixels per second throughput)
    df_eng["Display_Bandwidth_MP_s"] = (
        (df_eng["Total_Display_Pixels"] * df_eng["Refresh_Rate_Hz"]) / 1e6
    ).round(3)
    
    # 6. Silicon Density (Frequency per nm lithography efficiency)
    df_eng["Silicon_Density"] = (
        df_eng["Max_Core_Frequency_GHz"] / df_eng["Process_Technology_nm"].replace(0, 4)
    ).round(4)
    
    # 7. Total Cache Memory (KB across all core clusters)
    df_eng["Total_Cache_KB"] = (
        df_eng["Cache_Memory_Core1"].fillna(0) +
        df_eng["Cache_Memory_Core2"].fillna(0) +
        df_eng["Cache_Memory_Core3"].fillna(0)
    )
    
    # 8. Display Technology Category (AMOLED_OLED vs LCD)
    df_eng["Display_Tech_Category"] = df_eng["Display_Type"].apply(categorize_display_type)
    
    # 9. Extracted Rear Camera Megapixels
    df_eng["Rear_Camera_MP"] = df_eng["Camera_Specifications"].apply(extract_rear_camera_mp)
    
    # 10. Number of Active CPU Clusters (2 vs 3)
    df_eng["Num_CPU_Clusters"] = df_eng["Processor_Core3_Count"].apply(
        lambda count: 3 if count > 0 else 2
    )
    
    # 11. Memory Footprint Score (RAM x Base Storage)
    df_eng["Memory_Score"] = df_eng["RAM_Size_GB"] * df_eng["Storage_GB"]
    
    # 12. Compute Capability Composite (RAM * Max_Freq / Process_nm)
    df_eng["Compute_Capability_Score"] = (
        (df_eng["RAM_Size_GB"] * df_eng["Max_Core_Frequency_GHz"]) / df_eng["Process_Technology_nm"].replace(0, 4)
    ).round(3)
    
    # 13. SoC Tier and Manufacturer Categorization
    soc_tuples = df_eng["SoC"].apply(categorize_soc)
    df_eng["SoC_Tier"] = [t[0] for t in soc_tuples]
    df_eng["SoC_Maker"] = [t[1] for t in soc_tuples]
    
    return df_eng


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_cleaning import perform_data_cleaning
    from src.data_loader import load_raw_data
    df = load_raw_data()
    clean_df, _, _ = perform_data_cleaning(df)
    eng_df = engineer_features(clean_df)
    print("Feature engineering successful.")
    print(f"Engineered dataset shape: {eng_df.shape}")
