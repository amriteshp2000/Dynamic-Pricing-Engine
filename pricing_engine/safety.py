import numpy as np
import time
import math
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger("PriceEngine_Safety")

@dataclass
class SafetyConfig:
    min_price_global: float = 10.0       # Regulatory Floor
    max_price_global: float = 5000.0     # Regulatory Ceiling
    max_daily_change_pct: float = 0.25   # Stability
    min_margin_dollars: float = 10.0     # Profitability

@dataclass
class SafetyResult:
    safe_price: float
    original_price: float
    is_clamped: bool
    trigger_reason: str
    latency_ms: float

class SafetyLayer:
    """
    Module 04: Safety & Compliance Governor.
    Now Robust to NaN/Inf inputs.
    """
    
    def __init__(self, config: SafetyConfig = SafetyConfig()):
        self.config = config

    def validate_and_clamp(self, 
                           suggested_price: float, 
                           constraints: dict, 
                           prev_price: Optional[float] = None) -> SafetyResult:
        t0 = time.perf_counter()
        
        reasons = []
        original = suggested_price
        
        # --- 0. SANITIZATION (Critical Fix for B05) ---
        # Handle NaN, Inf, or None types immediately
        if suggested_price is None or not isinstance(suggested_price, (int, float)):
            suggested_price = 0.0 # Force to a number so we can clamp it up
            reasons.append("InvalidType")
            
        if math.isnan(suggested_price) or math.isinf(suggested_price):
            suggested_price = 0.0 # Treat garbage as $0 and let floor logic fix it
            reasons.append("NaN_or_Inf")
            
        final_p = suggested_price
        
        # --- 1. HOST & COST CONSTRAINTS ---
        host_min = constraints.get('min_price', 0)
        cost_floor = constraints.get('cost_basis', 0) + self.config.min_margin_dollars
        
        lower_bound = max(self.config.min_price_global, host_min, cost_floor)
        
        host_max = constraints.get('max_price', float('inf'))
        upper_bound = min(self.config.max_price_global, host_max)
        
        # Sanity Check for Inverted Bounds
        if lower_bound > upper_bound:
            upper_bound = max(upper_bound, lower_bound)
            reasons.append("InvertedBounds_Fixed")

        # Clamp 1: Hard Bounds
        if final_p < lower_bound:
            final_p = lower_bound
            reasons.append("Floor")
        elif final_p > upper_bound:
            final_p = upper_bound
            reasons.append("Ceiling")
            
        # --- 2. STABILITY CONSTRAINTS ---
        if prev_price is not None and prev_price > 0:
            delta_limit = prev_price * self.config.max_daily_change_pct
            stability_min = prev_price - delta_limit
            stability_max = prev_price + delta_limit
            
            # Clip stability range into Hard Bound range
            stability_min = max(stability_min, lower_bound)
            stability_max = min(stability_max, upper_bound)
            
            if final_p < stability_min:
                final_p = stability_min
                reasons.append("Smoothness_Drop")
            elif final_p > stability_max:
                final_p = stability_max
                reasons.append("Smoothness_Rise")

        t1 = time.perf_counter()
        latency = (t1 - t0) * 1000.0 
        
        return SafetyResult(
            safe_price=round(final_p, 2),
            original_price=original,
            is_clamped=(final_p != original),
            trigger_reason="|".join(reasons) if reasons else "None",
            latency_ms=latency
        )