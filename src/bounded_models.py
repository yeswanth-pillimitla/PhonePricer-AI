"""
Bounded Estimator Module
Provides domain-constrained linear regression models using scipy.optimize.lsq_linear.
Enforces non-negative or positive marginal sensitivity for physical hardware components
(specifically Battery_Capacity_mAh), preventing empirical negative partial coefficients
caused by observational collinearity with form factor / budget segment clustering.
"""

from typing import Optional, Union, List
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.base import BaseEstimator, RegressorMixin


class BoundedRidgeRegressor(RegressorMixin, BaseEstimator):
    """
    Ridge Regression with linear inequality constraints solved via bounded quadratic programming.
    Solves: min_w ||X w + b - y||_2^2 + alpha ||w||_2^2
    subject to: lb <= w <= ub
    """
    _estimator_type = "regressor"

    def __init__(
        self,
        alpha: float = 1.2,
        bat_lower_bound: float = 0.022,
        bat_idx: Optional[int] = 2
    ):
        self.alpha = alpha
        self.bat_lower_bound = bat_lower_bound
        self.bat_idx = bat_idx

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        
        self.mean_y_ = float(np.mean(y_arr))
        y_c = y_arr - self.mean_y_
        
        lb = np.full(n_features, -np.inf)
        ub = np.full(n_features, np.inf)
        
        # Enforce positive lower bound on battery capacity column
        target_idx = self.bat_idx if self.bat_idx is not None else 2
        if 0 <= target_idx < n_features:
            lb[target_idx] = self.bat_lower_bound
            
        A = np.vstack([X_arr, np.sqrt(self.alpha) * np.eye(n_features)])
        b_vec = np.concatenate([y_c, np.zeros(n_features)])
        
        res = lsq_linear(A, b_vec, bounds=(lb, ub))
        self.coef_ = res.x
        self.intercept_ = self.mean_y_
        self.n_features_in_ = n_features
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_ + self.intercept_


class BoundedHuberRegressor(RegressorMixin, BaseEstimator):
    """
    Huber M-estimator with bounded quadratic programming via Iteratively Reweighted Least Squares (IRLS).
    Combines outlier robustness with physics-respecting parameter lower bounds.
    """
    _estimator_type = "regressor"

    def __init__(
        self,
        alpha: float = 1.5,
        epsilon: float = 1.35,
        max_iter: int = 50,
        bat_lower_bound: float = 0.022,
        bat_idx: Optional[int] = 2
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.bat_lower_bound = bat_lower_bound
        self.bat_idx = bat_idx

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        
        self.mean_y_ = float(np.mean(y_arr))
        y_c = y_arr - self.mean_y_
        
        lb = np.full(n_features, -np.inf)
        ub = np.full(n_features, np.inf)
        
        target_idx = self.bat_idx if self.bat_idx is not None else 2
        if 0 <= target_idx < n_features:
            lb[target_idx] = self.bat_lower_bound
            
        w = np.zeros(n_features)
        weights = np.ones(n_samples)
        
        for _ in range(self.max_iter):
            W_sqrt = np.sqrt(weights)
            A = np.vstack([X_arr * W_sqrt[:, np.newaxis], np.sqrt(self.alpha) * np.eye(n_features)])
            b_vec = np.concatenate([y_c * W_sqrt, np.zeros(n_features)])
            
            res = lsq_linear(A, b_vec, bounds=(lb, ub))
            w_new = res.x
            
            residuals = y_c - X_arr @ w_new
            scale = np.median(np.abs(residuals)) / 0.6745 + 1e-6
            r = np.abs(residuals) / scale
            weights_new = np.where(r <= self.epsilon, 1.0, self.epsilon / np.maximum(r, 1e-6))
            
            if np.max(np.abs(w_new - w)) < 1e-4:
                w = w_new
                break
            w = w_new
            weights = weights_new
            
        self.coef_ = w
        self.intercept_ = self.mean_y_
        self.n_features_in_ = n_features
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_ + self.intercept_

