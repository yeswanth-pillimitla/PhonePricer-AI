"""
Hyperparameter Tuning Module
Executes GridSearchCV with 5-fold cross-validation on X_train.
Optimizes hyperparameters for Huber Regressor, SVR, Ridge, ElasticNet,
Random Forest, CatBoost, Gradient Boosting, Voting Ensembles, and Stacking Ensembles.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

from src.bounded_models import BoundedRidgeRegressor, BoundedHuberRegressor
from src.preprocessing import build_preprocessor
from src.model_evaluation import compute_metrics


def get_tuning_configurations() -> Dict[str, Dict[str, Any]]:
    """Define hyperparameter grids and base estimators."""
    return {
        "Bounded_Ridge": {
            "model": BoundedRidgeRegressor(bat_lower_bound=0.022),
            "scaling": "standard",
            "family": "Bounded_Linear",
            "params": {
                "regressor__regressor__alpha": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
            }
        },
        "Bounded_Huber": {
            "model": BoundedHuberRegressor(bat_lower_bound=0.022),
            "scaling": "standard",
            "family": "Bounded_Robust",
            "params": {
                "regressor__regressor__alpha": [1.0, 1.5, 2.0, 3.0, 5.0]
            }
        },
        "Huber": {
            "model": HuberRegressor(max_iter=2000),
            "scaling": "standard",
            "family": "Robust_Linear",
            "params": {
                "regressor__regressor__alpha": [3.0, 5.0, 7.0, 8.0, 10.0, 12.0, 15.0]
            }
        },
        "SVR": {
            "model": SVR(kernel="rbf"),
            "scaling": "standard",
            "family": "Support_Vector",
            "params": {
                "regressor__regressor__C": [5.0, 8.0, 10.0, 15.0, 20.0],
                "regressor__regressor__epsilon": [0.05, 0.08, 0.1, 0.15]
            }
        },
        "Ridge": {
            "model": Ridge(random_state=42),
            "scaling": "standard",
            "family": "Regularized_Linear",
            "params": {
                "regressor__regressor__alpha": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
            }
        },
        "ElasticNet": {
            "model": ElasticNet(random_state=42, max_iter=10000),
            "scaling": "standard",
            "family": "Regularized_Linear",
            "params": {
                "regressor__regressor__alpha": [0.005, 0.01, 0.02, 0.05, 0.1],
                "regressor__regressor__l1_ratio": [0.2, 0.5, 0.7]
            }
        },
        "Random_Forest": {
            "model": RandomForestRegressor(random_state=42),
            "scaling": None,
            "family": "Ensemble_Bagging",
            "params": {
                "regressor__regressor__n_estimators": [50, 100],
                "regressor__regressor__max_depth": [4, 6, None],
                "regressor__regressor__max_features": ["sqrt", 0.7]
            }
        },
        "CatBoost": {
            "model": CatBoostRegressor(random_seed=42, verbose=0),
            "scaling": None,
            "family": "Ensemble_Boosting",
            "params": {
                "regressor__regressor__iterations": [150, 200],
                "regressor__regressor__learning_rate": [0.03, 0.05],
                "regressor__regressor__depth": [3, 4]
            }
        },
        "Gradient_Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "scaling": None,
            "family": "Ensemble_Boosting",
            "params": {
                "regressor__regressor__n_estimators": [50, 100],
                "regressor__regressor__learning_rate": [0.03, 0.05],
                "regressor__regressor__max_depth": [2, 3]
            }
        }
    }


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Run hyperparameter tuning on training partition only.
    
    Returns:
        (tuned_pipelines_dict, best_params_dict, comparison_df)
    """
    configs = get_tuning_configurations()
    tuned_pipelines = {}
    best_params_dict = {}
    results = []
    
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for name, config in configs.items():
        preprocessor = build_preprocessor(num_cols, cat_cols, scaling=config["scaling"])
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", TransformedTargetRegressor(
                regressor=config["model"],
                func=np.log,
                inverse_func=np.exp
            ))
        ])
        
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=config["params"],
            cv=cv,
            scoring="r2",
            n_jobs=-1,
            refit=True
        )
        
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_
        tuned_pipelines[name] = best_pipe
        best_params_dict[name] = grid.best_params_
        
        # Predictions
        y_train_pred = best_pipe.predict(X_train)
        y_val_pred = best_pipe.predict(X_val)
        
        # Metrics
        train_metrics = compute_metrics(y_train, y_train_pred, dataset_name="")
        val_metrics = compute_metrics(y_val, y_val_pred, dataset_name="")
        
        # Cross-validation MAE and R2 on training partition
        cv_mae = -cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        cv_r2 = cross_val_score(best_pipe, X_train, y_train, cv=cv, scoring="r2")
        
        results.append({
            "Model_Name": f"{name} (Tuned)",
            "Family": config["family"],
            "Train_MAE": train_metrics["MAE"],
            "Train_RMSE": train_metrics["RMSE"],
            "Train_R2": train_metrics["R2"],
            "Val_MAE": val_metrics["MAE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
            "CV_MAE_Mean": round(float(np.mean(cv_mae)), 2),
            "CV_MAE_Std": round(float(np.std(cv_mae)), 2),
            "CV_R2_Mean": round(float(np.mean(cv_r2)), 4),
            "CV_R2_Std": round(float(np.std(cv_r2)), 4),
            "Overfitting_Gap_MAE": round(val_metrics["MAE"] - train_metrics["MAE"], 2)
        })
        
    # Also construct and evaluate Ensembles
    best_bhuber_alpha = best_params_dict.get("Bounded_Huber", {}).get("regressor__regressor__alpha", 1.5)
    best_bridge_alpha = best_params_dict.get("Bounded_Ridge", {}).get("regressor__regressor__alpha", 1.2)
    best_huber_alpha = best_params_dict.get("Huber", {}).get("regressor__regressor__alpha", 7.0)
    best_svr_c = best_params_dict.get("SVR", {}).get("regressor__regressor__C", 8.0)
    best_svr_eps = best_params_dict.get("SVR", {}).get("regressor__regressor__epsilon", 0.08)
    best_ridge_alpha = best_params_dict.get("Ridge", {}).get("regressor__regressor__alpha", 5.0)
    
    ensembles = {
        "Voting_Bounded_Huber_Ridge": VotingRegressor([
            ("bounded_huber", BoundedHuberRegressor(alpha=best_bhuber_alpha, bat_lower_bound=0.022)),
            ("bounded_ridge", BoundedRidgeRegressor(alpha=best_bridge_alpha, bat_lower_bound=0.022))
        ], weights=[0.50, 0.50]),
        "Voting_Huber_SVR": VotingRegressor([
            ("huber", BoundedHuberRegressor(alpha=best_bhuber_alpha, bat_lower_bound=0.022)),
            ("svr", SVR(C=best_svr_c, epsilon=best_svr_eps))
        ], weights=[0.75, 0.25]),
        "Voting_Huber_SVR_Ridge": VotingRegressor([
            ("huber", BoundedHuberRegressor(alpha=best_bhuber_alpha, bat_lower_bound=0.022)),
            ("svr", SVR(C=best_svr_c, epsilon=best_svr_eps)),
            ("ridge", BoundedRidgeRegressor(alpha=best_bridge_alpha, bat_lower_bound=0.022))
        ], weights=[0.70, 0.25, 0.05]),
        "Stacking_Ensemble": StackingRegressor([
            ("huber", BoundedHuberRegressor(alpha=best_bhuber_alpha, bat_lower_bound=0.022)),
            ("svr", SVR(C=best_svr_c, epsilon=best_svr_eps)),
            ("ridge", BoundedRidgeRegressor(alpha=best_bridge_alpha, bat_lower_bound=0.022))
        ], final_estimator=Ridge(alpha=1.0, random_state=42))
    }
    
    preprocessor = build_preprocessor(num_cols, cat_cols, scaling="standard")
    for ens_name, ens_estimator in ensembles.items():
        ens_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", TransformedTargetRegressor(
                regressor=ens_estimator,
                func=np.log,
                inverse_func=np.exp
            ))
        ])
        ens_pipe.fit(X_train, y_train)
        tuned_pipelines[ens_name] = ens_pipe
        best_params_dict[ens_name] = {"type": "ensemble"}
        
        y_train_pred = ens_pipe.predict(X_train)
        y_val_pred = ens_pipe.predict(X_val)
        
        train_metrics = compute_metrics(y_train, y_train_pred, dataset_name="")
        val_metrics = compute_metrics(y_val, y_val_pred, dataset_name="")
        
        cv_mae = -cross_val_score(ens_pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        cv_r2 = cross_val_score(ens_pipe, X_train, y_train, cv=cv, scoring="r2")
        
        results.append({
            "Model_Name": f"{ens_name} (Tuned)",
            "Family": "Ensemble_Meta",
            "Train_MAE": train_metrics["MAE"],
            "Train_RMSE": train_metrics["RMSE"],
            "Train_R2": train_metrics["R2"],
            "Val_MAE": val_metrics["MAE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],
            "CV_MAE_Mean": round(float(np.mean(cv_mae)), 2),
            "CV_MAE_Std": round(float(np.std(cv_mae)), 2),
            "CV_R2_Mean": round(float(np.mean(cv_r2)), 4),
            "CV_R2_Std": round(float(np.std(cv_r2)), 4),
            "Overfitting_Gap_MAE": round(val_metrics["MAE"] - train_metrics["MAE"], 2)
        })
        
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Val_MAE").reset_index(drop=True)
    return tuned_pipelines, best_params_dict, results_df


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
    
    tuned_pipes, best_params, results_df = tune_all_models(
        X_train, y_train, X_val, y_val, num_cols, cat_cols
    )
    print("Tuned Models Benchmark Results:")
    print(results_df[["Model_Name", "Val_MAE", "Val_R2", "CV_MAE_Mean", "CV_R2_Mean"]].to_string(index=False))
