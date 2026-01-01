import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold
from dataclasses import dataclass
from typing import Optional
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
    Module 01: Double Machine Learning (DML) for Causal Price Elasticity.

    Implements orthogonal learning:
    1. Residualize price:     T_res = T - E[T | X]
    2. Residualize demand:    Y_res = Y - E[Y | X]
    3. Estimate beta from:    Y_res = beta * T_res + eps

    All uncertainty is estimated via bootstrap.
    """

    def __init__(
        self,
        model_t: BaseEstimator = GradientBoostingRegressor(
            n_estimators=50, max_depth=3, subsample=0.7, random_state=42
        ),
        model_y: BaseEstimator = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, subsample=0.7, random_state=42
        ),
        n_splits: int = 3,
        n_bootstrap: int = 100,
        random_state: int = 42,
    ):
        self.model_t = model_t
        self.model_y = model_y
        self.n_splits = n_splits
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

        self.result: Optional[CausalResult] = None
        self.residuals_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Core DML estimation (single run)
    # ------------------------------------------------------------------
    def _fit_once(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> float:
        X = X.reset_index(drop=True)
        T = T.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        T_res = np.zeros(len(T))
        Y_res = np.zeros(len(Y))

        kf = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            T_train, T_test = T.iloc[train_idx], T.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

            # Fit nuisance models
            m_t = clone(self.model_t).fit(X_train, T_train)
            m_y = clone(self.model_y).fit(X_train, Y_train)

            # Residualize
            t_hat = m_t.predict(X_test)
            y_hat = m_y.predict_proba(X_test)[:, 1]

            T_res[test_idx] = T_test - t_hat
            Y_res[test_idx] = Y_test - y_hat

        # Save residuals from the *main* run
        self.residuals_ = pd.DataFrame(
            {"T_res": T_res, "Y_res": Y_res}
        )

        # Overlap check (hard guardrail)
        if np.percentile(np.abs(T_res), 1) < 1e-4:
            raise ValueError(
                "Insufficient overlap: residual price variation is near zero."
            )

        # Closed-form OLS slope
        beta = np.dot(T_res, Y_res) / np.dot(T_res, T_res)
        return beta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, T: pd.Series, Y: pd.Series) -> "DML_ElasticityModel":
        logger.info(
            f"Training DML with {len(X)} samples | "
            f"{self.n_splits}-fold CV | "
            f"{self.n_bootstrap} bootstrap draws"
        )

        rng = np.random.default_rng(self.random_state)
        n = len(X)

        betas = []

        for _ in range(self.n_bootstrap):
            sub_n = min(200_000, n)
            idx = rng.choice(n, sub_n, replace=True)
            beta = self._fit_once(
                X.iloc[idx], T.iloc[idx], Y.iloc[idx]
            )
            betas.append(beta)

        betas = np.array(betas)

        beta_hat = betas.mean()
        std_err = betas.std(ddof=1)
        ci_lower, ci_upper = np.percentile(betas, [2.5, 97.5])

        p_value = 2 * min(
            (betas <= 0).mean(),
            (betas >= 0).mean()
        )

        self.result = CausalResult(
            elasticity=beta_hat,
            std_err=std_err,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
        )

        logger.info(
            f"Causal Elasticity: {beta_hat:.4f} "
            f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        )

        return self
