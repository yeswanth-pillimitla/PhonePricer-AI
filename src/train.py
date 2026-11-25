"""
Model Training Module
Constructs, fits, and evaluates baseline dummy regressors and 16 regression algorithms:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Huber Regressor
6. Bayesian Ridge
7. Decision Tree Regressor
8. Random Forest Regressor
9. Extra Trees Regressor
10. Gradient Boosting Regressor
11. HistGradientBoosting Regressor
12. XGBoost Regressor
13. LightGBM Regressor
14. CatBoost Regressor
15. Support Vector Regression (SVR)
16. K-Nearest Neighbors Regressor (KNN)

Evaluates performance on Training, Validation, and 5-Fold Cross-Validation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score

from src.bounded_models import BoundedRidgeRegressor, BoundedHuberRegressor
from src.preprocessing import build_preprocessor
from src.model_evaluation import compute_metrics


def get_model_definitions() -> Dict[str, Tuple[Any, str, str, bool]]:
    """
    Define model registry with appropriate preprocessing scaling and log-target transform flag.
    Format: {model_name: (model_instance, scaling_type, family, use_log_target)}
    """
    return {
        "Dummy_Mean": (DummyRegressor(strategy="mean"), "standard", "Baseline", False),
        "Dummy_Median": (DummyRegressor(strategy="median"), "standard", "Baseline", False),
        "Linear_Regression": (LinearRegression(), "standard", "Linear", True),
        "Ridge": (Ridge(alpha=5.0, random_state=42), "standard", "Regularized_Linear", True),
        "Lasso": (Lasso(alpha=0.01, random_state=42), "standard", "Regularized_Linear", True),
        "ElasticNet": (ElasticNet(alpha=0.02, l1_ratio=0.5, random_state=42), "standard", "Regularized_Linear", True),
        "Huber": (HuberRegressor(alpha=7.0, max_iter=2000), "standard", "Robust_Linear", True),
        "Bounded_Ridge": (BoundedRidgeRegressor(alpha=1.2, bat_lower_bound=0.022), "standard", "Bounded_Linear", True),
        "Bounded_Huber": (BoundedHuberRegressor(alpha=1.5, bat_lower_bound=0.022), "standard", "Bounded_Robust", True),
        "Bayesian_Ridge": (BayesianRidge(), "standard", "Bayesian_Linear", True),
        "Decision_Tree": (DecisionTreeRegressor(max_depth=5, random_state=42), None, "Tree", True),
        "Random_Forest": (RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42), None, "Ensemble_Bagging", True),
        "Extra_Trees": (ExtraTreesRegressor(n_estimators=100, max_depth=6, random_state=42), None, "Ensemble_Bagging", True),
        "Gradient_Boosting": (GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42), None, "Ensemble_Boosting", True),
        "Hist_Gradient_Boosting": (HistGradientBoostingRegressor(learning_rate=0.05, max_depth=3, random_state=42), None, "Ensemble_Boosting", True),
        "XGBoost": (xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42), None, "Ensemble_Boosting", True),
        "LightGBM": (lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1), None, "Ensemble_Boosting", True),
        "CatBoost": (CatBoostRegressor(iterations=200, learning_rate=0.04, depth=4, random_seed=42, verbose=0), None, "Ensemble_Boosting", True),
        "SVR": (SVR(kernel="rbf", C=8.0, epsilon=0.08), "standard", "Support_Vector", True),
        "KNN": (KNeighborsRegressor(n_neighbors=5, weights="distance"), "standard", "Instance_Based", True)
    }


def train_and_evaluate_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
    """
    Train all baseline and candidate ML regression models.
    
    Returns:
        (fitted_pipelines_dict, comparison_df)
    """
    model_defs = get_model_definitions()
    fitted_pipelines = {}
    results = []
    
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for name, (model, scaling, family, use_log) in model_defs.items():
        preprocessor = build_preprocessor(num_cols, cat_cols, scaling=scaling)
        
        if use_log:
            pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp))
            ])
        else:
            pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ])
        
        # Fit on training partition only
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        # Predictions
        y_train_pred = pipe.predict(X_train)
        y_val_pred = pipe.predict(X_val)
        
        # Compute metrics
        train_metrics = compute_metrics(y_train, y_train_pred, dataset_name="")
        val_metrics = compute_metrics(y_val, y_val_pred, dataset_name="")
        
        # Cross-validation MAE and R2 on training set
        cv_mae_scores = -cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring="neg_mean_absolute_error"
        )
        cv_r2_scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring="r2"
        )
        
        results.append({
            "Model_Name": name,
            "Family": family,
            "Train_MAE": train_metrics["MAE"],
            "Train_RMSE": train_metrics["RMSE"],
            "Train_R2": train_metrics["R2"],
            "Val_MAE": val_metrics["MAE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
            "CV_MAE_Mean": round(float(np.mean(cv_mae_scores)), 2),
            "CV_MAE_Std": round(float(np.std(cv_mae_scores)), 2),
            "CV_R2_Mean": round(float(np.mean(cv_r2_scores)), 4),
            "CV_R2_Std": round(float(np.std(cv_r2_scores)), 4),
            "Overfitting_Gap_MAE": round(val_metrics["MAE"] - train_metrics["MAE"], 2)
        })
        
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Val_MAE").reset_index(drop=True)
    return fitted_pipelines, results_df


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.data_cleaning import perform_data_cleaning
    from src.feature_engineering import engineer_features
    from src.feature_selection import select_features
    from src.data_splitting import split_data
    
    raw_df = load_raw_data()
    clean_df, _, _ = perform_data_cleaning(raw_df)
    eng_df = engineer_features(clean_df)
    X, y, num_cols, cat_cols = select_features(eng_df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    fitted, results_df = train_and_evaluate_all_models(
        X_train, y_train, X_val, y_val, num_cols, cat_cols
    )
    print("Default Models Benchmark Results:")
    print(results_df[["Model_Name", "Val_MAE", "Val_R2", "CV_MAE_Mean", "CV_R2_Mean"]].to_string(index=False))
