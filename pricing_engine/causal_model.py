
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import scipy.stats as stats
from dataclasses import dataclass
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger("PriceEngine_Causal")

@dataclass
class CausalResult:
    elasticity: float
    std_err: float
    ci_lower: float
    ci_upper: float
    p_value: float

class DML_ElasticityModel:
    """
    Module 01: Double Machine Learning (DML) for Causal Inference.
    
    Implements Orthogonal Learning:
    1. Debias Price (T) using controls (X) -> Residual T_res
    2. Debias Demand (Y) using controls (X) -> Residual Y_res
    3. Regress Y_res ~ T_res to find Causal Elasticity (beta)
    """
    
    def __init__(self, 
                 model_t: BaseEstimator = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
                 model_y: BaseEstimator = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
                 n_splits: int = 3):
        self.model_t = model_t
        self.model_y = model_y
        self.n_splits = n_splits
        self.result: Optional[CausalResult] = None
        self.residuals_ = None  # Store for plotting

    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> 'DML_ElasticityModel':
        """
        Performs Cross-Fitting DML.
        X: Confounders (Seasonality, Neighborhood)
        T: Treatment (Log Price)
        Y: Outcome (Is_Booked)
        """
        logger.info(f"Training Causal Model on {len(X)} samples with {self.n_splits}-fold cross-fitting...")
        
        # Reset indices to ensure alignment
        X = X.reset_index(drop=True)
        T = T.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        # Arrays to store residuals
        T_res = np.zeros(len(T))
        Y_res = np.zeros(len(Y))
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # --- Cross-Fitting Loop ---
        for train_idx, test_idx in kf.split(X):
            # Split
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            T_train, T_test = T.iloc[train_idx], T.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
            
            # 1. Train Nuisance Models
            m_t = clone(self.model_t).fit(X_train, T_train)
            m_y = clone(self.model_y).fit(X_train, Y_train)
            
            # 2. Predict & Residualize
            # "What price did we expect given the season?"
            t_pred = m_t.predict(X_test)
            # "What demand did we expect given the season?"
            y_pred = m_y.predict(X_test)
            
            T_res[test_idx] = T_test - t_pred
            Y_res[test_idx] = Y_test - y_pred
            
        self.residuals_ = pd.DataFrame({'T_res': T_res, 'Y_res': Y_res})
        
        # --- Final Causal Regression (OLS on Residuals) ---
        # Y_res = beta * T_res + error
        # We use simple OLS logic here to get stats
        
        # 1. Coefficient (beta)
        # beta = cov(T_res, Y_res) / var(T_res)
        num = np.dot(T_res, Y_res)
        den = np.dot(T_res, T_res)
        beta = num / den
        
        # 2. Standard Error Calculation
        n = len(T_res)
        df = n - 1
        
        # Calculate residuals of the *final* regression
        final_epsilon = Y_res - beta * T_res
        mse = np.sum(final_epsilon**2) / df
        var_beta = mse / np.sum(T_res**2)
        std_err = np.sqrt(var_beta)
        
        # 3. Confidence Intervals (95%)
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = beta - t_crit * std_err
        ci_upper = beta + t_crit * std_err
        
        # 4. P-value
        t_stat = beta / std_err
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        self.result = CausalResult(
            elasticity=beta,
            std_err=std_err,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_val
        )
        
        logger.info(f"Causal Elasticity: {beta:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        return self