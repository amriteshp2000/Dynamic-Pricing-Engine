
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from .Bandit import ThompsonBandit
from .safety import SafetyLayer

@dataclass
class TrustMetrics:
    total_nights: int
    hist_revenue: float
    dr_revenue: float
    uplift_pct: float
    
    # Trust Signals
    override_rate: float        # % of prices differing >30% from host
    safety_clamp_rate: float    # % of prices modified by Safety Layer
    safety_violations: int      # Should be 0 (Critical Fail if >0)
    volatility_reduction: float # Comparison of std(price_diff)

class TrustEvaluator:
    """
    Module 05: Offline Evaluation & Trust Verification.
    
    Calculates Doubly Robust Revenue AND Behavioral Metrics.
    """
    
    def __init__(self, agent: ThompsonBandit, safety: SafetyLayer):
        self.agent = agent
        self.safety = safety

    def evaluate_trust(self, df_test: pd.DataFrame) -> TrustMetrics:
        """
        Replays history to generate a Trust Report.
        """
        n = len(df_test)
        
        # Trackers
        dr_revenues = []
        hist_revenues = []
        
        overrides = 0
        clamps = 0
        violations = 0
        
        agent_prices = []
        hist_prices = df_test['price'].tolist()
        
        # 1. Inference Loop
        # We process row by row to simulate the constraints context
        for idx, row in df_test.iterrows():
            context = row.to_dict()
            p_hist = row['price']
            y_hist = row['is_booked']
            
            # --- A. Constraint Inference ---
            # In prod, we pull from DB. Here, we infer from history for realism.
            # Floor = 50% of hist, Ceiling = 300% of hist
            constraints = {
                'min_price': p_hist * 0.5, 
                'max_price': p_hist * 3.0, 
                'cost_basis': p_hist * 0.4
            }
            
            # --- B. Agent Action ---
            # Bandit Suggestion
            decision = self.agent.choose_price(context)
            
            # Safety Governor
            safe_res = self.safety.validate_and_clamp(decision.selected_price, constraints)
            p_agent = safe_res.safe_price
            
            # --- C. Metrics Tracking ---
            
            # 1. Override Check (>30% deviation)
            deviation = abs(p_agent - p_hist) / p_hist
            if deviation > 0.30:
                overrides += 1
                
            # 2. Safety Clamp Check
            if safe_res.is_clamped:
                clamps += 1
                
            # 3. Violation Check (Critical)
            # Did we actually respect the constraints?
            eff_min = max(constraints['min_price'], constraints['cost_basis'] + 10)
            if p_agent < eff_min - 0.01 or p_agent > constraints['max_price'] + 0.01:
                violations += 1
            
            agent_prices.append(p_agent)
            
            # --- D. Doubly Robust Revenue Estimation ---
            # 1. Predict Revenue for Agent Price
            pred_agent = self.agent.prior_model.predict(context, p_agent)
            rev_agent_exp = p_agent * pred_agent.prob
            
            # 2. Predict Revenue for Historical Price (Bias Correction)
            pred_hist = self.agent.prior_model.predict(context, p_hist)
            rev_hist_exp = p_hist * pred_hist.prob
            
            # 3. DR Correction (Kernel Matching)
            # If historical price was close ($10), use the observed error
            if abs(p_hist - p_agent) < 10.0:
                observed_rev = p_hist * y_hist
                correction = observed_rev - rev_hist_exp
                dr_val = rev_agent_exp + correction
            else:
                dr_val = rev_agent_exp # Fallback to Direct Method
                
            dr_revenues.append(dr_val)
            hist_revenues.append(p_hist * y_hist)

        # 2. Aggregation
        total_hist = sum(hist_revenues)
        total_dr = sum(dr_revenues)
        uplift = (total_dr - total_hist) / total_hist if total_hist > 0 else 0
        
        # Volatility (Std Dev of day-to-day changes)
        # Simple proxy: Std Dev of prices
        vol_hist = np.std(hist_prices)
        vol_agent = np.std(agent_prices)
        vol_reduction = (vol_hist - vol_agent) / vol_hist if vol_hist > 0 else 0
        
        return TrustMetrics(
            total_nights=n,
            hist_revenue=total_hist,
            dr_revenue=total_dr,
            uplift_pct=uplift,
            override_rate=overrides / n,
            safety_clamp_rate=clamps / n,
            safety_violations=violations,
            volatility_reduction=vol_reduction
        )