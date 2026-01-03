import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Optional, Dict

# ML / Econ
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

import statsmodels.api as sm

from econml.grf import CausalForest

logger = logging.getLogger("PriceEngine_Causal")


# =====================================================================
# Shared result container
# =====================================================================
@dataclass
class CausalResult:
    model_name: str
    elasticity: float
    std_err: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    p_value: Optional[float]
    n_obs: int
    notes: Dict[str, float]


# =====================================================================
# Model 1: Fixed Effects OLS (Sanity / Falsification)
# =====================================================================
class FixedEffectElasticity:
    """
    Fixed Effects OLS on weekly panel data.

    ROLE:
        - Sanity check / falsification
        - NOT for deployment

    Model:
        Y_{iw} = beta * P_{iw} + alpha_i + gamma_w + eps
    """

    def fit(
        self,
        df: pd.DataFrame,
        price_col: str,
        outcome_col: str,
        listing_col: str,
        week_col: str,
    ) -> CausalResult:

        df = df.copy()

        # Two-way demeaning (listing + week)
        df["price_dm"] = (
            df[price_col]
            - df.groupby(listing_col)[price_col].transform("mean")
            - df.groupby(week_col)[price_col].transform("mean")
        )

        df["y_dm"] = (
            df[outcome_col]
            - df.groupby(listing_col)[outcome_col].transform("mean")
            - df.groupby(week_col)[outcome_col].transform("mean")
        )

        X = sm.add_constant(df["price_dm"])
        y = df["y_dm"]

        model = sm.OLS(y, X).fit()

        beta = model.params["price_dm"]
        se = model.bse["price_dm"]
        ci_low, ci_high = model.conf_int().loc["price_dm"]
        pval = model.pvalues["price_dm"]

        logger.info(
            f"[FixedEffects] Elasticity = {beta:.4f} "
            f"[{ci_low:.4f}, {ci_high:.4f}]"
        )

        return CausalResult(
            model_name="FixedEffectsOLS",
            elasticity=beta,
            std_err=se,
            ci_lower=ci_low,
            ci_upper=ci_high,
            p_value=pval,
            n_obs=len(df),
            notes={
                "role": "sanity_check_only",
            },
        )


# =====================================================================
# Model 2: Linear DML (Production Elasticity)
# =====================================================================
class LinearDMLElasticity:
    """
    Linear Double Machine Learning for weekly aggregated data.

    ROLE:
        - Production elasticity
        - Stable, auditable, deployable

    Assumes:
        - Each row is a (listing, week)
        - T = mean weekly price
        - Y = weekly booking indicator or rate
        - X = confounders
    """

    def __init__(
        self,
        model_t: Optional[BaseEstimator] = None,
        model_y: Optional[BaseEstimator] = None,
        n_splits: int = 3,
        n_bootstrap: int = 200,
        random_state: int = 42,
        min_overlap_std: float = 1e-3,
    ):
        self.model_t = model_t or GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            subsample=0.7,
            random_state=random_state,
        )
        self.model_y = model_y or GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            subsample=0.7,
            random_state=random_state,
        )

        self.n_splits = n_splits
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.min_overlap_std = min_overlap_std

        self.residuals_: Optional[pd.DataFrame] = None

    # ------------------------------
    # Single orthogonal run
    # ------------------------------
    def _fit_once(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        Y: pd.Series,
    ) -> float:

        T_res = np.zeros(len(T))
        Y_res = np.zeros(len(Y))

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for tr, te in kf.split(X):
            m_t = clone(self.model_t).fit(X.iloc[tr], T.iloc[tr])
            m_y = clone(self.model_y).fit(X.iloc[tr], Y.iloc[tr])

            T_res[te] = T.iloc[te] - m_t.predict(X.iloc[te])
            Y_res[te] = Y.iloc[te] - m_y.predict(X.iloc[te])

        # Overlap guardrail
        if np.std(T_res) < self.min_overlap_std:
            raise ValueError(
                "LinearDML overlap violation: "
                "residualized price has near-zero variance."
            )

        self.residuals_ = pd.DataFrame(
            {"T_res": T_res, "Y_res": Y_res}
        )

        return np.dot(T_res, Y_res) / np.dot(T_res, T_res)

    # ------------------------------
    # Public API
    # ------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        Y: pd.Series,
    ) -> CausalResult:

        rng = np.random.default_rng(self.random_state)
        n = len(X)
        betas = []

        logger.info(
            f"[LinearDML] Fitting | n={n} | "
            f"{self.n_splits}-fold | "
            f"{self.n_bootstrap} bootstrap"
        )

        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=min(n, 200_000), replace=True)
            beta = self._fit_once(
                X.iloc[idx],
                T.iloc[idx],
                Y.iloc[idx],
            )
            betas.append(beta)

        betas = np.asarray(betas)

        beta_hat = betas.mean()
        se = betas.std(ddof=1)
        ci_low, ci_high = np.percentile(betas, [2.5, 97.5])

        pval = 2 * min(
            (betas <= 0).mean(),
            (betas >= 0).mean(),
        )

        logger.info(
            f"[LinearDML] Elasticity = {beta_hat:.4f} "
            f"[{ci_low:.4f}, {ci_high:.4f}]"
        )

        return CausalResult(
            model_name="LinearDML",
            elasticity=beta_hat,
            std_err=se,
            ci_lower=ci_low,
            ci_upper=ci_high,
            p_value=pval,
            n_obs=n,
            notes={
                "role": "production",
                "bootstrap": self.n_bootstrap,
            },
        )

# =====================================================================
# Model 3: Causal Forest (Heterogeneity / Strategy)
# =====================================================================
# =====================================================================
# Model 3: Causal Forest (Heterogeneity / Strategy)
# =====================================================================
class CausalForestElasticity:
    """
    Causal Forest for heterogeneous elasticity discovery.

    ROLE:
        - Strategy / segmentation
        - Offline only (not deployed directly)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        min_samples_leaf: int = 50,
        random_state: int = 42,
    ):
        # We assume _ECONML_AVAILABLE is checked globally or imported safely
        self.model = CausalForest(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        Y: pd.Series,
    ) -> CausalResult:

        # 1. Fit the model (Ensure y is passed as 'y', not 'Y')
        self.model.fit(
            X=X.values,
            T=T.values,
            y=Y.values,
        )

        # 2. FIX: Use .predict() instead of .effect()
        # predict() returns the Conditional Average Treatment Effect (theta)
        effects = self.model.predict(X.values)
        
        # If output is (N, 1), flatten it to (N,) for cleaner stats
        if effects.ndim > 1 and effects.shape[1] == 1:
            effects = effects.flatten()

        logger.info(
            f"[CausalForest] Mean elasticity = {effects.mean():.4f} "
            f"(std={effects.std():.4f})"
        )

        return CausalResult(
            model_name="CausalForest",
            elasticity=float(effects.mean()),
            # Note: Here std_err reflects the heterogeneity (spread) of effects, 
            # not the standard error of the mean estimate.
            std_err=float(effects.std()),
            ci_lower=float(np.percentile(effects, 5)),
            ci_upper=float(np.percentile(effects, 95)),
            p_value=None,
            n_obs=len(X),
            notes={
                "role": "heterogeneity_analysis",
                "heterogeneity_std": float(effects.std()),
            },
        )


# =====================================================================
# Orchestrator: Causal Triangulation Runner
# =====================================================================
def run_causal_pipeline(
    df_weekly: pd.DataFrame,
    feature_cols: list,
    price_col: str = "avg_price",
    outcome_col: str = "is_booked",
    listing_col: str = "listing_id",
    week_col: str = "week_date",
) -> Dict[str, CausalResult]:
    """
    Executes the full causal triangulation strategy on weekly panel data.

    Sequence:
        1. Fixed Effects OLS   (Sanity / Falsification)
        2. Linear DML          (Production Elasticity)
        3. Causal Forest       (Heterogeneity / Strategy)

    Parameters
    ----------
    df_weekly : pd.DataFrame
        Weekly aggregated panel. Each row = (listing_id, week).
    feature_cols : list
        Columns used as confounders X.
    price_col : str
        Weekly mean price column.
    outcome_col : str
        Weekly booking indicator or rate.
    listing_col : str
        Listing identifier.
    week_col : str
        Weekly time identifier.

    Returns
    -------
    Dict[str, CausalResult]
        Results from each causal model.
    """

    results: Dict[str, CausalResult] = {}

    # -----------------------------------------------------------------
    # 1. Model 1: Fixed Effects OLS (Sanity Check)
    # -----------------------------------------------------------------
    logger.info(">>> Running Model 1: Fixed Effects OLS (Sanity Check)")
    fe_model = FixedEffectElasticity()

    res_fe = fe_model.fit(
        df=df_weekly,
        price_col=price_col,
        outcome_col=outcome_col,
        listing_col=listing_col,
        week_col=week_col,
    )
    results["sanity_check"] = res_fe

    # Hard warning (not a hard stop, but very serious)
    if res_fe.elasticity > 0:
        logger.warning(
            "!!! CRITICAL WARNING !!!\n"
            "Fixed Effects sanity check shows POSITIVE elasticity.\n"
            "This suggests severe confounding or mis-specified aggregation.\n"
            "Proceeding further is NOT recommended without investigation."
        )

    # -----------------------------------------------------------------
    # 2. Model 2: Linear DML (Production Model)
    # -----------------------------------------------------------------
    logger.info(">>> Running Model 2: Linear DML (Production Elasticity)")

    X = df_weekly[feature_cols].fillna(0)

    # Log-price is standard for elasticity interpretation
    T = np.log1p(df_weekly[price_col])

    # Binary or rate outcome (weekly)
    Y = df_weekly[outcome_col]

    dml_model = LinearDMLElasticity(
        n_bootstrap=50  # keep modest for iteration; increase for final runs
    )
    res_dml = dml_model.fit(X, T, Y)
    results["production"] = res_dml

    # -----------------------------------------------------------------
    # 3. Model 3: Causal Forest (Strategy / Heterogeneity)
    # -----------------------------------------------------------------
    
    logger.info(">>> Running Model 3: Causal Forest (Heterogeneity / Strategy)")
    cf_model = CausalForestElasticity()
    res_cf = cf_model.fit(X, T, Y)
    results["strategy"] = res_cf
    

    return results
