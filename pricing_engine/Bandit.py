import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable, Tuple
import logging

logger = logging.getLogger("PriceEngine_Bandit")

__all__ = [
    "PricingDecision",
    "ThompsonPricingBandit",
    "BayesianUCBBandit",
    "LinUCBBandit",
]

# ============================================================
# Output schema
# ============================================================
@dataclass
class PricingDecision:
    selected_price: float
    expected_revenue: float
    uncertainty_sigma: float
    source: str
    panic_mode: bool


# ============================================================
# Utilities
# ============================================================
def sigmoid(z):
    z = np.clip(z, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


# ============================================================
# Online Bayesian Logistic (fast + stable)
# ============================================================
class StreamingBayesianLogistic:
    """
    Numerically stable online Bayesian logistic regression
    using precision-matrix updates with forgetting and jitter.
    """

    def __init__(self, n_features: int, lambda_forget=0.95, prior_var=5.0):
        self.lam = lambda_forget
        self.mu = np.zeros(n_features)

        # Precision matrix (inverse covariance)
        self.P = np.eye(n_features) / prior_var
        self.min_precision = 1e-3  # CRITICAL SAFETY FLOOR

        self.n_updates = 0

    def update(self, x: np.ndarray, y: float):
        x = x.reshape(-1)

        # Forgetting (but keep SPD)
        self.P = self.lam * self.P + (1 - self.lam) * np.eye(len(x)) * self.min_precision

        p = sigmoid(self.mu @ x)

        # Curvature floor (prevents zero Hessian)
        W = max(p * (1 - p), 1e-3)

        # Rank-1 precision update
        self.P += W * np.outer(x, x)

        # Safe Newton step
        try:
            delta = np.linalg.solve(self.P, x * (y - p))
        except np.linalg.LinAlgError:
            # Emergency stabilization (should almost never trigger)
            self.P += np.eye(len(x)) * self.min_precision
            delta = np.linalg.solve(self.P, x * (y - p))

        self.mu += delta
        self.n_updates += 1

    def predict(self, x: np.ndarray):
        x = x.reshape(-1)
        mean = sigmoid(self.mu @ x)
        mean = np.clip(mean, 0.001, 0.999)

        # Compute variance via solve (no inverse)
        v = np.linalg.solve(self.P, x)
        var_latent = x @ v
        var_prob = (mean * (1 - mean))**2 * var_latent

        return mean, np.sqrt(var_prob + 1e-9)

# ============================================================
# Base Bandit (STRICT PRODUCTION CONTRACT)
# ============================================================
class BasePricingBandit:
    """
    Model-agnostic pricing bandit.

    The bandit only knows:
        predict_fn(X) -> (mean, std)

    It NEVER inspects demand models.
    """

    def __init__(
        self,
        *,
        feature_names,
        scaler,
        predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        forgetting_factor=0.95,
        min_updates_to_trust=5,
        panic_threshold=3,
    ):
        self.feature_names = feature_names
        self.scaler = scaler
        self.predict_fn = predict_fn

        self.lam = forgetting_factor
        self.min_updates = min_updates_to_trust
        self.panic_threshold = panic_threshold

        self.online_models: Dict[str, StreamingBayesianLogistic] = {}
        self.failures: Dict[str, int] = {}

    # -------------------------
    # Feature handling
    # -------------------------
    def _context_vec(self, context: dict) -> np.ndarray:
        x_raw = np.array([[context[f] for f in self.feature_names]])
        return self.scaler.transform(x_raw).flatten()

    def _full_vec_batch(self, x_ctx: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """
        Vectorized feature construction:
        [context_features, log1p(price)]
        """
        return np.column_stack([
            np.repeat(x_ctx.reshape(1, -1), len(prices), axis=0),
            np.log1p(prices)
        ])

    # -------------------------
    # Panic logic (safety)
    # -------------------------
    def _panic_adjust(self, lid, min_p, max_p):
        fails = self.failures.get(lid, 0)
        panic = fails >= self.panic_threshold
        eff_max = max_p * (0.85 ** max(0, fails - self.panic_threshold + 1))
        return panic, max(min_p, eff_max)

    # -------------------------
    # Prior prediction (black box)
    # -------------------------
    def _prior_predict_batch(self, X: np.ndarray):
        mean, std = self.predict_fn(X)
        mean = np.clip(mean, 0.001, 0.999)
        std = np.maximum(std, 1e-6)
        return mean, std


# ============================================================
# 1️⃣ Thompson Sampling Bandit
# ============================================================
class ThompsonPricingBandit(BasePricingBandit):

    def update(self, context, price, booked):
        lid = str(context["listing_id"])
        x_ctx = self._context_vec(context)
        x = np.hstack([x_ctx, np.log1p(price)])

        if lid not in self.online_models:
            self.online_models[lid] = StreamingBayesianLogistic(len(x), self.lam)
            self.failures[lid] = 0

        self.online_models[lid].update(x, float(booked))
        self.failures[lid] = self.failures[lid] + 1 if booked == 0 else 0

    def choose_price(self, context, min_p=50, max_p=300, n_candidates=40):
        lid = str(context["listing_id"])
        panic, eff_max = self._panic_adjust(lid, min_p, max_p)
        prices = np.linspace(min_p, eff_max, n_candidates)

        x_ctx = self._context_vec(context)
        X = self._full_vec_batch(x_ctx, prices)

        # Prior
        mean_p, std_p = self._prior_predict_batch(X)

        # Online fusion
        if lid in self.online_models:
            online = self.online_models[lid]
            w = min(1.0, online.n_updates / self.min_updates)

            mean_o = np.array([online.predict(x)[0] for x in X])
            std_o = np.array([online.predict(x)[1] for x in X])

            means = (1 - w) * mean_p + w * mean_o
            stds = np.sqrt((1 - w) * std_p**2 + w * std_o**2)
        else:
            means, stds = mean_p, std_p

        samples = np.clip(
            means + stds * np.random.randn(len(means)),
            0.01,
            0.99,
        )

        revenues = prices * samples
        idx = int(np.argmax(revenues))

        return PricingDecision(
            selected_price=prices[idx],
            expected_revenue=prices[idx] * means[idx],
            uncertainty_sigma=stds[idx],
            source="Panic" if panic else "Thompson",
            panic_mode=panic,
        )


# ============================================================
# 2️⃣ Bayesian UCB Bandit
# ============================================================
class BayesianUCBBandit(BasePricingBandit):

    def __init__(self, *args, beta=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def update(self, context, price, booked):
        lid = str(context["listing_id"])
        x_ctx = self._context_vec(context)
        x = np.hstack([x_ctx, np.log1p(price)])

        if lid not in self.online_models:
            self.online_models[lid] = StreamingBayesianLogistic(len(x), self.lam)
            self.failures[lid] = 0

        self.online_models[lid].update(x, float(booked))
        self.failures[lid] = self.failures[lid] + 1 if booked == 0 else 0

    def choose_price(self, context, min_p=50, max_p=300, n_candidates=40):
        lid = str(context["listing_id"])
        panic, eff_max = self._panic_adjust(lid, min_p, max_p)
        prices = np.linspace(min_p, eff_max, n_candidates)

        x_ctx = self._context_vec(context)
        X = self._full_vec_batch(x_ctx, prices)

        mean_p, std_p = self._prior_predict_batch(X)

        if lid in self.online_models:
            online = self.online_models[lid]
            w = min(1.0, online.n_updates / self.min_updates)

            mean_o = np.array([online.predict(x)[0] for x in X])
            std_o = np.array([online.predict(x)[1] for x in X])

            means = (1 - w) * mean_p + w * mean_o
            stds = np.sqrt((1 - w) * std_p**2 + w * std_o**2)
        else:
            means, stds = mean_p, std_p

        scores = prices * (means + self.beta * stds)
        idx = int(np.argmax(scores))

        return PricingDecision(
            selected_price=prices[idx],
            expected_revenue=prices[idx] * means[idx],
            uncertainty_sigma=stds[idx],
            source="Panic" if panic else "BayesianUCB",
            panic_mode=panic,
        )


# ============================================================
# 3️⃣ LinUCB (fast deterministic baseline)
# ============================================================
class LinUCBBandit(BasePricingBandit):

    def __init__(self, *args, alpha=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.A = {}
        self.A_inv = {}
        self.b = {}

    def update(self, context, price, booked):
        lid = str(context["listing_id"])
        x_ctx = self._context_vec(context)
        x = np.hstack([x_ctx, np.log1p(price)])

        if lid not in self.A:
            d = len(x)
            self.A[lid] = np.eye(d)
            self.A_inv[lid] = np.eye(d)
            self.b[lid] = np.zeros(d)
            self.failures[lid] = 0

        A_inv = self.A_inv[lid]
        Ax = A_inv @ x
        denom = max(1.0 + x @ Ax, 1e-6)

        self.A_inv[lid] = A_inv - np.outer(Ax, Ax) / denom
        self.A[lid] += np.outer(x, x)
        self.b[lid] += booked * x
        self.failures[lid] = self.failures[lid] + 1 if booked == 0 else 0

    def choose_price(self, context, min_p=50, max_p=300, n_candidates=40):
        lid = str(context["listing_id"])
        panic, eff_max = self._panic_adjust(lid, min_p, max_p)
        prices = np.linspace(min_p, eff_max, n_candidates)

        x_ctx = self._context_vec(context)
        X = self._full_vec_batch(x_ctx, prices)

        if lid in self.A_inv:
            theta = self.A_inv[lid] @ self.b[lid]
            means = sigmoid(X @ theta)
            sigmas = np.sqrt(np.sum(X @ self.A_inv[lid] * X, axis=1))
        else:
            means, sigmas = self._prior_predict_batch(X)

        scores = prices * (means + self.alpha * sigmas)
        idx = int(np.argmax(scores))

        return PricingDecision(
            selected_price=prices[idx],
            expected_revenue=prices[idx] * means[idx],
            uncertainty_sigma=sigmas[idx],
            source="Panic" if panic else "LinUCB",
            panic_mode=panic,
        )
