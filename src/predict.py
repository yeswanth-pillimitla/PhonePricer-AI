"""
Prediction and Model Persistence Module
Handles saving and loading the complete production ML artifact bundle:
- Fitted Preprocessing Pipeline + Estimator
- Selected Feature Manifests
- Hyperparameter Configuration & Evaluation Metrics
- Robust Inference API with validation and automatic feature derivation
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Union, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.feature_engineering import engineer_features
from src.feature_selection import SELECTED_NUMERICAL_FEATURES, SELECTED_CATEGORICAL_FEATURES


def format_inr(val: Union[int, float], symbol: str = "₹") -> str:
    """Format a number into the Indian numbering system (e.g. ₹15,000, ₹1,25,000)."""
    val = int(round(val))
    is_neg = val < 0
    s = str(abs(val))
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.append(rest)
        parts.reverse()
        formatted = ",".join(parts) + "," + last3
    prefix = f"{symbol}" if symbol else ""
    return f"-{prefix}{formatted}" if is_neg else f"{prefix}{formatted}"


def save_model_bundle(
    pipeline: Pipeline,
    model_name: str,
    best_params: Dict[str, Any],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    output_path: str = "models/best_smartphone_price_model.joblib"
) -> str:
    """
    Serialize the complete pipeline along with production metadata.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    bundle = {
        "pipeline": pipeline,
        "model_name": model_name,
        "best_params": best_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "numerical_features": SELECTED_NUMERICAL_FEATURES,
        "categorical_features": SELECTED_CATEGORICAL_FEATURES,
        "target": "Price_INR",
        "currency": "INR",
        "currency_symbol": "₹",
        "training_records": 77,  # Train (63) + Val (14)
        "test_records": 14
    }
    
    joblib.dump(bundle, output_path)
    return output_path


def load_model_bundle(model_path: str = "models/best_smartphone_price_model.joblib") -> Dict[str, Any]:
    """Load serialized model bundle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model bundle not found at: {model_path}")
    return joblib.load(model_path)


def predict_smartphone_price(
    specifications: Union[Dict[str, Any], pd.DataFrame],
    model_bundle: Optional[Dict[str, Any]] = None,
    model_path: str = "models/best_smartphone_price_model.joblib"
) -> Dict[str, Any]:
    """
    Predict smartphone price in BDT from hardware specifications.
    
    Args:
        specifications: Dict or DataFrame containing phone specifications.
        model_bundle: Pre-loaded model bundle (optional).
        model_path: Path to serialized model artifact if bundle not supplied.
        
    Returns:
        Dict containing predicted price, error margin (based on test MAE),
        model name, and formatted details.
    """
    if model_bundle is None:
        model_bundle = load_model_bundle(model_path)
        
    pipeline = model_bundle["pipeline"]
    model_name = model_bundle["model_name"]
    test_mae = model_bundle.get("test_metrics", {}).get("Test_MAE", 6500.0)
    
    # Convert dict to DataFrame if needed
    if isinstance(specifications, dict):
        df_input = pd.DataFrame([specifications])
    else:
        df_input = specifications.copy()
        
    # Set default values for any omitted raw columns
    defaults = {
        "Phone_Company": "Xiaomi Redmi",
        "Phone_Model": "Custom Smartphone Spec",
        "SoC": "Standard Octa-Core SoC",
        "Processor_Core_Count": 8,
        "Processor_Core1": "Cortex-A78",
        "Processor_Core1_Count": 2,
        "Processor_Core1_Frequency_GHz": 2.4,
        "Processor_Core2": "Cortex-A55",
        "Processor_Core2_Count": 6,
        "Processor_Core2_Frequency_GHz": 2.0,
        "Processor_Core3": "Not Applicable",
        "Processor_Core3_Count": 0,
        "Processor_Core3_Frequency_GHz": 0.0,
        "RAM_Type": "LPDDR4X",
        "RAM_Size_GB": 8,
        "Cache_Memory_Core1": 512,
        "Cache_Memory_Core2": 128,
        "Cache_Memory_Core3": 0,
        "Storage_GB": 128,
        "Battery_Capacity_mAh": 5000,
        "ISA": "Armv8.2-A",
        "Frequency_of_Different_Cores": "2x2.4 GHz; 6x2.0 GHz",
        "Process_Technology_nm": 4,
        "Display_Type": "AMOLED, 120Hz",
        "Display_Resolution": "1080 x 2400 pixels",
        "Display_Resolution_Width_px": 1080,
        "Display_Resolution_Height_px": 2400,
        "Refresh_Rate_Hz": 120,
        "Camera_Specifications": "Rear: 50 MP, f/1.8 | Front: 16 MP"
    }
    
    for k, v in defaults.items():
        if k not in df_input.columns:
            df_input[k] = v
            
    # Apply feature engineering to derive computed attributes
    df_engineered = engineer_features(df_input)
    
    # Filter to selected feature set
    selected_cols = SELECTED_NUMERICAL_FEATURES + SELECTED_CATEGORICAL_FEATURES
    X_pred = df_engineered[selected_cols]
    
    # Predict directly in INR (no currency conversion)
    raw_prediction = pipeline.predict(X_pred)
    predicted_price = float(np.round(raw_prediction[0], -1))  # round to nearest 10 INR
    
    # Ensure reasonable lower bound (no negative prices)
    predicted_price = max(predicted_price, 5000.0)
    
    # Estimated realistic price range based on test MAE
    lower_estimate = max(predicted_price - test_mae, 4500.0)
    upper_estimate = predicted_price + test_mae
    
    return {
        "predicted_price_inr": predicted_price,
        "predicted_price": predicted_price,
        "predicted_price_bdt": predicted_price,  # backwards compatibility alias
        "formatted_price": format_inr(predicted_price),
        "expected_range_inr": (round(lower_estimate, -1), round(upper_estimate, -1)),
        "expected_range_bdt": (round(lower_estimate, -1), round(upper_estimate, -1)),
        "formatted_range": f"{format_inr(lower_estimate)} - {format_inr(upper_estimate)}",
        "currency": "INR",
        "currency_symbol": "₹",
        "model_name": model_name,
        "confidence_note": (
            f"Estimated price based on {model_name}. Historical test MAE is ±{format_inr(test_mae)}. "
            "Market retail prices in India may vary with local taxes, retailer promotions, and warranty terms."
        )
    }


if __name__ == "__main__":
    sample_phone = {
        "Phone_Company": "OnePlus",
        "RAM_Type": "LPDDR5X",
        "RAM_Size_GB": 12,
        "Storage_GB": 256,
        "Battery_Capacity_mAh": 6000,
        "Process_Technology_nm": 3,
        "Display_Resolution_Width_px": 1264,
        "Display_Resolution_Height_px": 2780,
        "Refresh_Rate_Hz": 120,
        "Display_Type": "AMOLED, 120Hz",
        "Processor_Core1_Count": 1,
        "Processor_Core1_Frequency_GHz": 3.3,
        "Processor_Core2_Count": 5,
        "Processor_Core2_Frequency_GHz": 3.1,
        "Processor_Core3_Count": 2,
        "Processor_Core3_Frequency_GHz": 2.3,
        "Cache_Memory_Core1": 2048,
        "Cache_Memory_Core2": 512,
        "Cache_Memory_Core3": 256,
        "ISA": "Armv9.2-A",
        "Camera_Specifications": "Rear: 50 MP, PDAF, OIS | Front: 32 MP"
    }
    
    print("Testing prediction function with sample flagship spec...")
    # Will be tested after model is saved in main pipeline

