import time
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import pandas as pd
import numpy as np

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
    # New Config for Lattice
    lattice_min_prob: float = 0.005  
    lattice_hallucination_thresh: float = 3.0

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
        if price is None:
            reasons.append("InvalidType")
            return 0.0
        # Handle numpy floats
        try:
            price = float(price)
        except:
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
# MODEL-SPECIFIC SAFETY (UPDATED WITH LATTICE)
# ---------------------------------------------------------------------

class ModelSafetyPolicy:
    """
    Penalizes aggressive prices when model uncertainty is high.
    Also enforces Monotonicity using TF_Lattice.
    """
    def __init__(self, config: SafetyConfig):
        self.config = config

    def apply_uncertainty_damp(self, price: float, demand_meta: Dict, reasons: list) -> float:
        std = max(demand_meta.get("std_dev", 0.0), 0.0)
        if std > 0.0:
            damp = 1.0 - min(std * self.config.uncertainty_penalty_pct, 0.5)
            # Only damp if price is above average (prevent dumping price on uncertainty)
            # Simplified for now: always damp towards conservative baseline
            reasons.append("Uncertainty_Damp")
            return price * damp
        return price
    
    def apply_lattice_physics(
        self, 
        price: float, 
        predicted_prob: float,
        context_row: pd.Series, 
        models: Dict[str, Any], 
        reasons: list
    ) -> float:
        """
        New Logic: Checks TF_Lattice for hallucination.
        """
        safety_model = models.get("TF_Lattice")
        if not safety_model:
            return price
            
        # 1. Probe
        probe = pd.DataFrame([context_row])
        probe["log_price"] = np.log1p(price)
        # Fix categories if needed
        if hasattr(safety_model, "cat_cols"):
             for c in safety_model.cat_cols:
                 if c in probe.columns: probe[c] = probe[c].astype("category")

        try:
            safety_prob = safety_model.predict(probe)[0]
        except:
            return price # Fail open if model fails
            
        # 2. Check Hallucination
        # If Bandit says 15% booking probability, but Lattice says 1%
        if predicted_prob > (safety_prob * self.config.lattice_hallucination_thresh) and \
           safety_prob < self.config.lattice_min_prob:
            
            reasons.append("Lattice_Physics_Clamp")
            # Return a safer price (e.g., 90% of current)
            return price * 0.9
            
        return price

# ---------------------------------------------------------------------
# BANDIT-SPECIFIC SAFETY
# ---------------------------------------------------------------------

class BanditSafetyPolicy:
    def __init__(self, config: SafetyConfig):
        self.config = config

    def apply(self, price: float, prev_price: Optional[float], reasons: list) -> float:
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
    Universal safety governor.
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
        predicted_prob: float,  # NEW: Needed for Lattice check
        context_row: pd.Series, # NEW: Needed for Lattice check
        demand_models: Dict,    # NEW: Needed for Lattice check
        constraints: dict = {},
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
        p = self.model_policy.apply_uncertainty_damp(p, demand_meta, reasons)
        
        # 2. Lattice Physics Check (NEW)
        p = self.model_policy.apply_lattice_physics(p, predicted_prob, context_row, demand_models, reasons)

        # 3. Hard regulatory & cost bounds
        p = self.base.hard_bounds(p, constraints, reasons)

        # 4. Bandit smoothness constraint
        p = self.bandit_policy.apply(p, prev_price, reasons)

        # 5. Re-apply hard bounds (Safety MUST dominate smoothness)
        p = self.base.hard_bounds(p, constraints, reasons)

        latency = (time.perf_counter() - t0) * 1000.0

        return SafetyResult(
            safe_price=round(p, 2),
            original_price=original,
            is_clamped=(abs(p - original) > EPS),
            trigger_reason="|".join(dict.fromkeys(reasons)) if reasons else "None",
            latency_ms=latency
        )