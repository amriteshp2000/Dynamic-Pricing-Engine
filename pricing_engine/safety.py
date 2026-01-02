import time
import math
from dataclasses import dataclass
from typing import Optional, Dict
import logging

logger = logging.getLogger("PriceEngine_Safety")

EPS = 1e-6

# ---------------------------------------------------------------------
# CONFIGS
# ---------------------------------------------------------------------

@dataclass
class SafetyConfig:
    min_price_global: float = 10.0
    max_price_global: float = 5000.0
    max_daily_change_pct: float = 0.25
    min_margin_dollars: float = 10.0
    uncertainty_penalty_pct: float = 0.15


@dataclass
class SafetyResult:
    safe_price: float
    original_price: float
    is_clamped: bool
    trigger_reason: str
    latency_ms: float


# ---------------------------------------------------------------------
# BASE SAFETY (ALWAYS APPLIED)
# ---------------------------------------------------------------------

class BaseSafety:
    def __init__(self, config: SafetyConfig):
        self.config = config

    def sanitize(self, price: float, reasons: list) -> float:
        if price is None or not isinstance(price, (int, float)):
            reasons.append("InvalidType")
            return 0.0
        if math.isnan(price) or math.isinf(price):
            reasons.append("NaN_or_Inf")
            return 0.0
        return price

    def hard_bounds(self, price: float, constraints: dict, reasons: list) -> float:
        floor = max(
            self.config.min_price_global,
            constraints.get("min_price", 0.0),
            constraints.get("cost_basis", 0.0) + self.config.min_margin_dollars
        )

        ceiling = min(
            self.config.max_price_global,
            constraints.get("max_price", float("inf"))
        )

        if floor > ceiling:
            reasons.append("InvertedBounds_Fixed")
            ceiling = floor

        if price < floor:
            reasons.append("Floor")
            return floor

        if price > ceiling:
            reasons.append("Ceiling")
            return ceiling

        return price


# ---------------------------------------------------------------------
# MODEL-SPECIFIC SAFETY
# ---------------------------------------------------------------------

class ModelSafetyPolicy:
    """
    Penalizes aggressive prices when model uncertainty is high.
    """

    def __init__(self, config: SafetyConfig):
        self.config = config

    def apply(self, price: float, demand_meta: Dict, reasons: list) -> float:
        std = max(demand_meta.get("std_dev", 0.0), 0.0)

        if std > 0.0:
            damp = 1.0 - min(std * self.config.uncertainty_penalty_pct, 0.5)
            reasons.append("Uncertainty_Damp")
            return price * damp

        return price


# ---------------------------------------------------------------------
# BANDIT-SPECIFIC SAFETY
# ---------------------------------------------------------------------

class BanditSafetyPolicy:
    """
    Enforces smooth price evolution across bandit rounds.
    """

    def __init__(self, config: SafetyConfig):
        self.config = config

    def apply(
        self,
        price: float,
        prev_price: Optional[float],
        reasons: list
    ) -> float:

        if prev_price is None or prev_price <= 0:
            return price

        delta = prev_price * self.config.max_daily_change_pct
        min_p = prev_price - delta
        max_p = prev_price + delta

        if price < min_p:
            reasons.append("Smoothness_Drop")
            return min_p

        if price > max_p:
            reasons.append("Smoothness_Rise")
            return max_p

        return price


# ---------------------------------------------------------------------
# MASTER SAFETY GOVERNOR
# ---------------------------------------------------------------------

class SafetyGovernor:
    """
    Universal safety governor for all bandits and demand models.
    Deterministic, O(1), and benchmark-safe.
    """

    def __init__(self, config: SafetyConfig = SafetyConfig()):
        self.config = config
        self.base = BaseSafety(config)
        self.model_policy = ModelSafetyPolicy(config)
        self.bandit_policy = BanditSafetyPolicy(config)

    def validate_and_clamp(
        self,
        suggested_price: float,
        constraints: dict,
        prev_price: Optional[float] = None,
        demand_meta: Optional[dict] = None
    ) -> SafetyResult:

        t0 = time.perf_counter()
        reasons = []
        original = suggested_price
        demand_meta = demand_meta or {}

        # 0. Sanitization
        p = self.base.sanitize(suggested_price, reasons)

        # 1. Model-aware uncertainty damping
        p = self.model_policy.apply(p, demand_meta, reasons)

        # 2. Hard regulatory & cost bounds (authoritative)
        p = self.base.hard_bounds(p, constraints, reasons)

        # 3. Bandit smoothness constraint
        p = self.bandit_policy.apply(p, prev_price, reasons)

        # 4. Re-apply hard bounds to dominate all effects
        p = self.base.hard_bounds(p, constraints, reasons)

        latency = (time.perf_counter() - t0) * 1000.0

        return SafetyResult(
            safe_price=round(p, 2),
            original_price=original,
            is_clamped=(
                original is None
                or not isinstance(original, (int, float))
                or abs(p - original) > EPS
            ),
            trigger_reason="|".join(dict.fromkeys(reasons)) if reasons else "None",
            latency_ms=latency
        )
