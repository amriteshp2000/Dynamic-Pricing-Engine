import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict
from sklearn.preprocessing import StandardScaler
import logging
from pygam import LogisticGAM, s, f

logger = logging.getLogger("PriceEngine_Demand")

# ---------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------
@dataclass
class DemandPrediction:
    prob: float
    std_dev: float
    epistemic_std: float
    aleatoric_std: float
    source_level: str
    n_obs: int


# ---------------------------------------------------------------------
# Bayesian Logistic Regression via IRLS (Laplace Approximation)
# ---------------------------------------------------------------------
class BayesianLogisticIRLS:
    """
    Bayesian Logistic Regression with Gaussian prior
    using Iteratively Reweighted Least Squares (Laplace approximation).
    """

    def __init__(self, prior_mean: np.ndarray, prior_cov: np.ndarray, max_iter: int = 25):
        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.prior_prec = np.linalg.inv(prior_cov)
        self.max_iter = max_iter
        self.posterior_mean = None
        self.posterior_cov = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLogisticIRLS":
        w = self.prior_mean.copy()

        for _ in range(self.max_iter):
            z = X @ w
            p = self._sigmoid(z)

            W = np.clip(p * (1.0 - p), 1e-6, None)
            z_tilde = z + (y - p) / W

            H = X.T @ (W[:, None] * X) + self.prior_prec
            b = X.T @ (W * z_tilde) + self.prior_prec @ self.prior_mean

            w_new = np.linalg.solve(H, b)
            if np.linalg.norm(w_new - w) < 1e-6:
                break
            w = w_new

        self.posterior_mean = w
        self.posterior_cov = np.linalg.inv(H)
        return self

    def predict(self, X: np.ndarray) -> tuple[float, float, float, float]:
        mu = X @ self.posterior_mean
        var = np.sum(X @ self.posterior_cov * X, axis=1)

        prob = self._sigmoid(mu / np.sqrt(1.0 + np.pi * var / 8.0))
        prob = np.clip(prob, 1e-6, 1.0 - 1e-6)

        epistemic = np.sqrt(var)
        aleatoric = np.sqrt(prob * (1 - prob))
        predictive = np.sqrt(epistemic**2 + aleatoric**2)

        return float(prob[0]), float(predictive[0]), float(epistemic[0]), float(aleatoric[0])


# ---------------------------------------------------------------------
# Hierarchical Bayesian Logistic Demand Model (PRIMARY)
# ---------------------------------------------------------------------
class HierarchicalBayesianLogisticDemand:
    """
    PRIMARY DEMAND MODEL

    P(Y = 1 | x, p) = σ(wᵀx + β · log(p))

    Hierarchy:
        City → Neighborhood → Listing
    """

    def __init__(self, beta_price: float, min_listing_obs: int = 10):
        self.beta = beta_price
        self.min_listing_obs = min_listing_obs

        self.city_model = None
        self.neighborhood_models: Dict[str, BayesianLogisticIRLS] = {}
        self.listing_models: Dict[str, BayesianLogisticIRLS] = {}

        self.city_n = 0
        self.neighborhood_n: Dict[str, int] = {}
        self.listing_n: Dict[str, int] = {}

        self.feature_cols = None
        self.scaler = StandardScaler()

    def _design(self, X: np.ndarray, log_price: np.ndarray) -> np.ndarray:
        return np.hstack([X, self.beta * log_price])

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        logger.info("Training Hierarchical Bayesian Logistic Demand Model")

        self.feature_cols = feature_cols
        self.city_n = len(df)

        X = self.scaler.fit_transform(df[feature_cols].values)
        log_price = np.log1p(df["price"].values).reshape(-1, 1)
        y = df["is_booked"].values

        X_full = self._design(X, log_price)
        dim = X_full.shape[1]

        # City model
        self.city_model = BayesianLogisticIRLS(
            prior_mean=np.zeros(dim),
            prior_cov=np.eye(dim) * 5.0
        ).fit(X_full, y)

        # Neighborhood models
        for hood, g in df.groupby("neighborhood"):
            if len(g) < 20:
                continue
            model = BayesianLogisticIRLS(
                self.city_model.posterior_mean,
                self.city_model.posterior_cov
            ).fit(X_full[g.index], y[g.index])

            self.neighborhood_models[hood] = model
            self.neighborhood_n[hood] = len(g)

        # Listing models
        for lid, g in df.groupby("listing_id"):
            if len(g) < self.min_listing_obs:
                continue
            hood = g["neighborhood"].iloc[0]
            prior = self.neighborhood_models.get(hood, self.city_model)

            model = BayesianLogisticIRLS(
                prior.posterior_mean,
                prior.posterior_cov
            ).fit(X_full[g.index], y[g.index])

            self.listing_models[str(lid)] = model
            self.listing_n[str(lid)] = len(g)

        logger.info("Hierarchy training complete")

    def predict(self, context: dict, price: float) -> DemandPrediction:
        X = self.scaler.transform(
            np.array([[context[f] for f in self.feature_cols]])
        )
        X_full = self._design(X, np.array([[np.log1p(price)]]))

        lid = str(context.get("listing_id"))
        hood = context.get("neighborhood")

        if lid in self.listing_models:
            model, level, n_obs = self.listing_models[lid], "listing", self.listing_n[lid]
        elif hood in self.neighborhood_models:
            model, level, n_obs = self.neighborhood_models[hood], "neighborhood", self.neighborhood_n[hood]
        else:
            model, level, n_obs = self.city_model, "city", self.city_n

        prob, std, epi, ale = model.predict(X_full)

        return DemandPrediction(prob, std, epi, ale, level, n_obs)


# ---------------------------------------------------------------------
# Seasonal Elasticity Extension
# ---------------------------------------------------------------------
class SeasonalElasticityDemand(HierarchicalBayesianLogisticDemand):
    """
    Adds season × price interaction using contextual month.
    Month is treated as a relative seasonal indicator (scaled).
    """

    def _design(self, X: np.ndarray, log_price: np.ndarray) -> np.ndarray:
        """
        X[:, 0] is assumed to be 'month' (scaled).
        """
        month_scaled = X[:, 0:1]      # shape (1, 1)
        seasonal = month_scaled * self.beta * log_price
        return np.hstack([X, self.beta * log_price, seasonal])


# ---------------------------------------------------------------------
# Neighborhood Residual Corrector (Bias Repair)
# ---------------------------------------------------------------------
class NeighborhoodResidualCorrector:
    def __init__(self, shrinkage: float = 0.1):
        self.residual_means: Dict[str, float] = {}
        self.shrinkage = shrinkage

    def fit(self, df: pd.DataFrame, model) -> None:
        for hood, g in df.groupby("neighborhood"):
            preds = np.array([
                model.predict(r.to_dict(), r["price"]).prob
                for _, r in g.iterrows()
            ])
            self.residual_means[hood] = self.shrinkage * float(
                np.mean(g["is_booked"].values - preds)
            )

    def adjust(self, prob: float, neighborhood: str) -> float:
        return float(np.clip(prob + self.residual_means.get(neighborhood, 0.0), 0.0, 1.0))


# ---------------------------------------------------------------------
# Monotonicity Enforcer (Safety Layer)
# ---------------------------------------------------------------------
class MonotoneDemandWrapper:
    """
    SAFETY LAYER ONLY.
    Enforces monotonicity of demand w.r.t price.
    """

    def __init__(self, base_model):
        self.base_model = base_model

    def predict_curve(self, context: dict, price_grid: np.ndarray) -> np.ndarray:
        preds = np.array([self.base_model.predict(context, p).prob for p in price_grid])
        return np.minimum.accumulate(preds[::-1])[::-1]


# ---------------------------------------------------------------------
# Competing Model: Monotone GAM (Benchmark)
# ---------------------------------------------------------------------
class MonotoneGAMDemand:
    """
    Benchmark model only.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        X = self.scaler.fit_transform(df[feature_cols].values)
        y = df["is_booked"].values

        self.model = LogisticGAM(
            s(0) + f(1) + f(2) + s(3, constraints="monotonic_dec")
        ).fit(X, y)

    def predict(self, context: dict, price: float) -> DemandPrediction:
        X = self.scaler.transform(np.array([[
            context["day_of_year"],
            context["dow"],
            context["neighborhood_enc"],
            np.log1p(price)
        ]]))

        prob = float(self.model.predict_proba(X)[0])
        std = float(np.sqrt(prob * (1 - prob)))

        return DemandPrediction(prob, std, 0.0, std, "GAM", n_obs=0)
