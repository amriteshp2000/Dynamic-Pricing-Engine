
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger("PriceEngine_Safety")

@dataclass
class SafetyConfig:
    min_price_global: float = 10.0       # Regulatory Floor
    max_price_global: float = 5000.0     # Regulatory Ceiling (Anti-Gouging)
    max_daily_change_pct: float = 0.25   # Stability: Max +/- 25% per day
    min_margin_dollars: float = 10.0     # Profitability: Cost + $10

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
    
    Enforces a strict hierarchy of constraints:
    1. Hard Floors/Ceilings (Global & Listing specific)
    2. Profitability (Cost Basis)
    3. Stability (Delta Limits)
    
    Performance Goal: < 1ms
    """
    
    def __init__(self, config: SafetyConfig = SafetyConfig()):
        self.config = config

    def validate_and_clamp(self, 
                           suggested_price: float, 
                           constraints: dict, 
                           prev_price: Optional[float] = None) -> SafetyResult:
        """
        Projects the Bandit's suggestion into the Safe Set.
        
        constraints = {
            'min_price': float,  (Host Preference)
            'max_price': float,  (Host Preference)
            'cost_basis': float  (Cleaning + Fixed Costs)
        }
        """
        t0 = time.perf_counter()
        
        final_p = suggested_price
        reasons = []
        
        # --- 1. HOST & COST CONSTRAINTS (Hard) ---
        # The 'Effective Floor' is the highest of: Global Min, Host Min, or Cost+Margin
        host_min = constraints.get('min_price', 0)
        cost_floor = constraints.get('cost_basis', 0) + self.config.min_margin_dollars
        
        lower_bound = max(self.config.min_price_global, host_min, cost_floor)
        
        # The 'Effective Ceiling' is the lowest of: Global Max, Host Max
        host_max = constraints.get('max_price', float('inf'))
        upper_bound = min(self.config.max_price_global, host_max)
        
        # Sanity Check: If bound is inverted (Min > Max), prefer Min (Safety First)
        if lower_bound > upper_bound:
            # Fallback: Just use the cost floor
            upper_bound = max(upper_bound, lower_bound)
            reasons.append("InvertedBounds_Fixed")

        # Clamp 1: Hard Bounds
        if final_p < lower_bound:
            final_p = lower_bound
            reasons.append("Floor")
        elif final_p > upper_bound:
            final_p = upper_bound
            reasons.append("Ceiling")
            
        # --- 2. STABILITY CONSTRAINTS (Soft/Smoothed) ---
        # Only apply if we have a previous price
        if prev_price is not None:
            delta_limit = prev_price * self.config.max_daily_change_pct
            stability_min = prev_price - delta_limit
            stability_max = prev_price + delta_limit
            
            # We clip to stability bounds, BUT we must respect Hard Bounds (Hierarchy)
            # So we clip stability range *into* the Hard Bound range first
            stability_min = max(stability_min, lower_bound)
            stability_max = min(stability_max, upper_bound)
            
            if final_p < stability_min:
                final_p = stability_min
                reasons.append("Smoothness_Drop")
            elif final_p > stability_max:
                final_p = stability_max
                reasons.append("Smoothness_Rise")

        t1 = time.perf_counter()
        latency = (t1 - t0) * 1000.0 # to ms
        
        return SafetyResult(
            safe_price=round(final_p, 2),
            original_price=suggested_price,
            is_clamped=(final_p != suggested_price),
            trigger_reason="|".join(reasons) if reasons else "None",
            latency_ms=latency
        )