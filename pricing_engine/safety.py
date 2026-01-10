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
    # Additional logic bounds
    cost_basis_multiplier: float = 1.1

@dataclass
class SafetyResult:
    safe_price: float
    original_price: float
    is_clamped: bool
    is_valid: bool
    trigger_reason: str
    latency_ms: float
    reasons: list 

# ---------------------------------------------------------------------
# BASE SAFETY 
# ---------------------------------------------------------------------

class BaseSafety:
    def __init__(self, config: SafetyConfig):
        self.config = config

    def sanitize(self, price: float, reasons: list) -> float:
        if price is None:
            reasons.append("InvalidType")
            return 0.0
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
            reasons.append(f"Floor({floor:.2f})")
            return floor
        if price > ceiling:
            reasons.append(f"Ceiling({ceiling:.2f})")
            return ceiling
        return price

# ---------------------------------------------------------------------
# MODEL-SPECIFIC SAFETY (SCHEMA-AWARE CASTING)
# ---------------------------------------------------------------------

class ModelSafetyPolicy:
    def __init__(self, config: SafetyConfig):
        self.config = config

    def apply_uncertainty_damp(self, price: float, demand_meta: Dict, reasons: list) -> float:
        std = max(demand_meta.get("std_dev", 0.0), 0.0)
        if std > 0.0:
            damp = 1.0 - min(std * self.config.uncertainty_penalty_pct, 0.5)
            reasons.append("Uncertainty_Damp")
            return price * damp
        return price
    
    def _safe_predict(self, model, row_dict, price, reasons, model_name="SafetyModel"):
        """
        Helper to robustly predict using a FastKerasWrapper or Keras model.
        Forces types based on the model's feature list.
        """
        try:
            # 1. Identify Model Features
            # FastKerasWrapper stores them in .features
            # Standard Keras model might have .input_names
            features = []
            if hasattr(model, "features"):
                features = model.features
            elif hasattr(model, "input_names"):
                features = model.input_names
            
            # If we can't find features, we can't sanitize efficiently. 
            # Fallback to simple DataFrame construction (risky but necessary fallback)
            if not features:
                features = list(row_dict.keys())

            # 2. Build Typed Dictionary
            # We explicitly check "is this categorical?"
            clean_dict = {}
            
            # Known categoricals (Hardcoded fallback if metadata missing)
            known_cats = ["neighborhood", "room_type"]
            if hasattr(model, "cat_cols"):
                known_cats = model.cat_cols

            # Inject the current price we are testing
            row_dict["log_price"] = np.log1p(price)
            row_dict["avg_price"] = price 
            
            for feat in features:
                val = row_dict.get(feat, 0.0) # Default 0 if missing
                
                if feat in known_cats:
                    # Keep as string/category
                    clean_dict[feat] = str(val)
                else:
                    # Force Float
                    try:
                        clean_dict[feat] = float(val)
                    except:
                        clean_dict[feat] = 0.0

            # 3. Create DataFrame with explicit schema
            df = pd.DataFrame([clean_dict])
            
            # 4. Final Type Enforcement on DataFrame
            for col in df.columns:
                if col in known_cats:
                    df[col] = df[col].astype("category")
                else:
                    df[col] = df[col].astype("float32") # TF prefers float32

            # 5. Predict
            pred = model.predict(df)
            
            # Unpack result
            if isinstance(pred, (np.ndarray, list)):
                return float(pred[0])
            return float(pred)

        except Exception as e:
            # Log error but don't crash
            # reasons.append(f"{model_name}_Error({str(e)})") # Optional verbose logging
            return None

    def apply_lattice_physics(
        self, 
        price: float, 
        predicted_prob: float,
        context_row: Optional[pd.Series], 
        models: Optional[Dict[Any, Any]], 
        reasons: list
    ) -> float:
        if (models is None) or (context_row is None):
            return price

        # Find Safety Model
        safety_model = None
        for key in models:
            if "Lattice" in str(key) or "SAFETY" in str(key):
                safety_model = models[key]
                break
        
        if not safety_model:
            return price
            
        # Prepare Data Dict
        if isinstance(context_row, pd.Series):
            data_dict = context_row.to_dict()
        else:
            data_dict = dict(context_row)

        # Predict Safety Score
        safety_prob = self._safe_predict(safety_model, data_dict, price, reasons, "Lattice")
        
        if safety_prob is None:
            return price

        # Check Hallucination
        threshold_prob = float(self.config.lattice_min_prob)
        
        if (predicted_prob > (safety_prob * self.config.lattice_hallucination_thresh)) and \
           (safety_prob < threshold_prob):
            reasons.append(f"Lattice_Physics_Clamp(SafeProb={safety_prob:.4f})")
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
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config if config else SafetyConfig()
        self.base = BaseSafety(self.config)
        self.model_policy = ModelSafetyPolicy(self.config)
        self.bandit_policy = BanditSafetyPolicy(self.config)

    def validate_and_clamp(
        self,
        price: float,
        constraints: dict = {},
        context_row: Optional[pd.Series] = None,
        demand_models: Optional[Dict[Any, Any]] = None,
        prev_price: Optional[float] = None,
        demand_meta: Optional[dict] = None
    ) -> SafetyResult:

        t0 = time.perf_counter()
        reasons = []
        original = price
        demand_meta = demand_meta or {}

        # 0. Sanitization
        p = self.base.sanitize(price, reasons)

        # 1. Model-aware uncertainty damping
        p = self.model_policy.apply_uncertainty_damp(p, demand_meta, reasons)
        
        # 2. Lattice Physics Check
        predicted_prob = 0.0
        if demand_models and context_row is not None:
             # Find Mean Model
             mean_model = None
             for k in demand_models:
                 if "MEAN" in str(k) or "LGBM" in str(k):
                     mean_model = demand_models[k]
                     break
             
             if mean_model:
                 # Safe Prediction for Mean Model
                 if isinstance(context_row, pd.Series):
                    d_dict = context_row.to_dict()
                 else:
                    d_dict = dict(context_row)
                 
                 # We predict prob at the current price
                 prob = self.model_policy._safe_predict(mean_model, d_dict, p, reasons, "MeanModel")
                 if prob is not None:
                     predicted_prob = prob

        p = self.model_policy.apply_lattice_physics(p, predicted_prob, context_row, demand_models, reasons)

        # 3. Hard bounds
        p = self.base.hard_bounds(p, constraints, reasons)

        # 4. Smoothness
        p = self.bandit_policy.apply(p, prev_price, reasons)

        # 5. Re-apply hard bounds
        p = self.base.hard_bounds(p, constraints, reasons)

        latency = (time.perf_counter() - t0) * 1000.0
        
        if p <= 0.0:
            p = constraints.get('min_price', self.config.min_price_global)
            reasons.append("Zero_Rescue")

        return SafetyResult(
            safe_price=round(p, 2),
            original_price=original,
            is_clamped=(abs(p - original) > EPS),
            is_valid=True,
            trigger_reason="|".join(dict.fromkeys(reasons)) if reasons else "None",
            latency_ms=latency,
            reasons=reasons
        )