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
    source_level: str


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

            W = p * (1.0 - p)
            W = np.clip(W, 1e-6, None)

            # IRLS working response
            z_tilde = z + (y - p) / W

            # Hessian + prior
            H = X.T @ (W[:, None] * X) + self.prior_prec
            b = X.T @ (W * z_tilde) + self.prior_prec @ self.prior_mean

            w_new = np.linalg.solve(H, b)

            if np.linalg.norm(w_new - w) < 1e-6:
                break
            w = w_new

        self.posterior_mean = w
        self.posterior_cov = np.linalg.inv(H)
        return self

    def predict(self, X: np.ndarray) -> tuple[float, float]:
        mu = X @ self.posterior_mean
        var = np.sum(X @ self.posterior_cov * X, axis=1)

        # Logistic-Gaussian approximation
        prob = self._sigmoid(mu / np.sqrt(1.0 + np.pi * var / 8.0))
        prob = np.clip(prob, 1e-6, 1.0 - 1e-6)

        return float(prob[0]), float(np.sqrt(var[0]))


# ---------------------------------------------------------------------
# Hierarchical Bayesian Logistic Demand Model (PRIMARY)
# ---------------------------------------------------------------------
class HierarchicalBayesianLogisticDemand:
    """
    PRIMARY DEMAND MODEL

    P(Y = 1 | x, p) = σ(wᵀx + β · log(p))

    Hierarchy:
        City → Neighborhood → Listing

    β is fixed from Module 01 (causal elasticity).
    """

    def __init__(self, beta_price: float, min_listing_obs: int = 10):
        self.beta = beta_price
        self.min_listing_obs = min_listing_obs

        self.city_model: BayesianLogisticIRLS | None = None
        self.neighborhood_models: Dict[str, BayesianLogisticIRLS] = {}
        self.listing_models: Dict[str, BayesianLogisticIRLS] = {}

        self.feature_cols: list[str] | None = None
        self.scaler = StandardScaler()

    def _design(self, X: np.ndarray, log_price: np.ndarray) -> np.ndarray:
        return np.hstack([X, self.beta * log_price])

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        logger.info("Training Hierarchical Bayesian Logistic Demand Model")

        self.feature_cols = feature_cols

        # Feature preparation
        X_raw = df[feature_cols].values
        X = self.scaler.fit_transform(X_raw)

        log_price = np.log1p(df["price"].values).reshape(-1, 1)
        y = df["is_booked"].values

        X_full = self._design(X, log_price)
        dim = X_full.shape[1]

        # -------- City level --------
        city_prior_mean = np.zeros(dim)
        city_prior_cov = np.eye(dim) * 5.0

        self.city_model = BayesianLogisticIRLS(city_prior_mean, city_prior_cov)
        self.city_model.fit(X_full, y)

        # -------- Neighborhood level --------
        for hood, g in df.groupby("neighborhood"):
            if len(g) < 20:
                continue

            prior = self.city_model
            model = BayesianLogisticIRLS(prior.posterior_mean, prior.posterior_cov)
            model.fit(X_full[g.index], y[g.index])
            self.neighborhood_models[hood] = model

        # -------- Listing level --------
        for lid, g in df.groupby("listing_id"):
            if len(g) < self.min_listing_obs:
                continue

            hood = g["neighborhood"].iloc[0]
            prior = self.neighborhood_models.get(hood, self.city_model)

            model = BayesianLogisticIRLS(prior.posterior_mean, prior.posterior_cov)
            model.fit(X_full[g.index], y[g.index])
            self.listing_models[str(lid)] = model

        logger.info("Hierarchy training complete")

    def predict(self, context: dict, price: float) -> DemandPrediction:
        X_raw = np.array([[context[f] for f in self.feature_cols]])
        X = self.scaler.transform(X_raw)

        log_price = np.log1p(price)
        X_full = self._design(X, np.array([[log_price]]))

        lid = str(context.get("listing_id"))
        hood = context.get("neighborhood")

        if lid in self.listing_models:
            model, level = self.listing_models[lid], "listing"
        elif hood in self.neighborhood_models:
            model, level = self.neighborhood_models[hood], "neighborhood"
        else:
            model, level = self.city_model, "city"

        prob, std = model.predict(X_full)
        return DemandPrediction(prob=prob, std_dev=std, source_level=level)


# ---------------------------------------------------------------------
# Seasonal Price Elasticity Extension
# ---------------------------------------------------------------------
class SeasonalElasticityDemand(HierarchicalBayesianLogisticDemand):
    """
    Adds season × price interaction term.
    Assumes `month` is feature index 0.
    """

    def _design(self, X: np.ndarray, log_price: np.ndarray) -> np.ndarray:
        month = X[:, 0:1]
        seasonal_term = month * self.beta * log_price
        return np.hstack([X, self.beta * log_price, seasonal_term])


# ---------------------------------------------------------------------
# Neighborhood Residual Corrector (Lightweight)
# ---------------------------------------------------------------------
class NeighborhoodResidualCorrector:
    """
    Lightweight post-hoc residual smoother per neighborhood.
    """

    def __init__(self):
        self.residual_means: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> None:
        for hood, g in df.groupby("neighborhood"):
            self.residual_means[hood] = g["is_booked"].mean()

    def adjust(self, prob: float, neighborhood: str) -> float:
        return float(
            np.clip(prob + 0.05 * self.residual_means.get(neighborhood, 0.0), 0.0, 1.0)
        )


# ---------------------------------------------------------------------
# Monotonicity Enforcer (Advanced Safety Wrapper)
# ---------------------------------------------------------------------
class MonotoneDemandWrapper:
    """
    Enforces monotonicity of demand w.r.t price.
    """

    def __init__(self, base_model: HierarchicalBayesianLogisticDemand):
        self.base_model = base_model

    def predict_curve(self, context: dict, price_grid: np.ndarray) -> np.ndarray:
        preds = np.array(
            [self.base_model.predict(context, p).prob for p in price_grid]
        )
        return np.minimum.accumulate(preds[::-1])[::-1]


# ---------------------------------------------------------------------
# Competing Model: Monotone GAM Demand (Benchmark Only)
# ---------------------------------------------------------------------


class MonotoneGAMDemand:
    """
    Competing demand learner for benchmarking.

    Properties:
    - Fast
    - Semi-parametric
    - Enforces monotonic demand ↓ price
    - No hierarchy
    - No causal claims

    Used ONLY for robustness comparison.
    """

    def __init__(self):
        self.model: LogisticGAM | None = None
        self.feature_cols: list[str] | None = None
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        """
        Fit monotone GAM.

        Expected feature order:
            [day_of_year, dow, neighborhood_enc, log_price]
        """
        self.feature_cols = feature_cols

        X_raw = df[feature_cols].values
        X = self.scaler.fit_transform(X_raw)
        y = df["is_booked"].values

        # GAM specification:
        # smooth(day_of_year)
        # factor(dow)
        # factor(neighborhood)
        # monotone smooth(log_price)
        self.model = LogisticGAM(
            s(0) +                # day_of_year
            f(1) +                # dow
            f(2) +                # neighborhood_enc
            s(3, constraints="monotonic_dec")  # log(price)
        )

        self.model.fit(X, y)

    def predict(self, context: dict, price: float) -> DemandPrediction:
        """
        Predict demand probability for a given price.
        """
        X_raw = np.array([[
            context["day_of_year"],
            context["dow"],
            context["neighborhood_enc"],
            np.log1p(price)
        ]])

        X = self.scaler.transform(X_raw)
        prob = float(self.model.predict_proba(X)[0])

        # GAM uncertainty is approximated (not Bayesian)
        std = float(
            np.sqrt(
                prob * (1 - prob)
            )
        )

        return DemandPrediction(
            prob=np.clip(prob, 1e-6, 1 - 1e-6),
            std_dev=std,
            source_level="GAM"
        )
