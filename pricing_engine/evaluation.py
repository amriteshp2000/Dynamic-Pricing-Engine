# pricing_engine/evaluation.py

import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass
import logging

# UPDATED IMPORT: Assumes you renamed pricing_agent.py -> bandit.py
from .Bandit import ThompsonBandit  
from .safety import SafetyLayer

logger = logging.getLogger("PriceEngine_Eval")

@dataclass
class EvalMetric:
    policy_name: str
    total_revenue_est: float
    uplift_pct: float
    confidence_interval: Tuple[float, float]

class DoublyRobustEvaluator:
    """
    Module 05: Offline Policy Evaluator.
    Uses Doubly Robust (DR) Estimation to compare New Policy vs Historical Policy.
    
    The Equation:
    V_dr = V_direct + (V_observed - V_direct_hist) * w
    
    Where we correct the model's prediction using real observed errors 
    when the historical price matches our new price.
    """
    
    def __init__(self, agent: ThompsonBandit, safety: SafetyLayer):
        self.agent = agent
        self.safety = safety

    def evaluate_episode(self, df_history: pd.DataFrame) -> EvalMetric:
        """
        Replays the historical episode day-by-day.
        df_history must have: [context_features..., 'price' (historical), 'is_booked' (historical)]
        """
        n = len(df_history)
        
        # 1. Historical Revenue (Baseline)
        # What the human hosts actually made
        hist_revenue = (df_history['price'] * df_history['is_booked']).sum()
        
        # 2. Policy Simulation
        dr_values = []
        
        for idx, row in df_history.iterrows():
            context = row.to_dict()
            p_hist = row['price']
            y_hist = row['is_booked']
            rev_hist = p_hist * y_hist
            
            # --- A. Agent Decision (Counterfactual) ---
            # We assume the agent sees the same context
            decision = self.agent.choose_price(context)
            
            # --- B. Safety Clamp ---
            # We infer constraints from history for the simulation
            # (In prod, these come from a DB)
            constraints = {
                'min_price': p_hist * 0.5, 
                'max_price': p_hist * 3.0, 
                'cost_basis': p_hist * 0.4
            }
            safe_res = self.safety.validate_and_clamp(decision.selected_price, constraints)
            p_new = safe_res.safe_price
            
            # --- C. Model Predictions ---
            # 1. Predict Outcome for NEW Price (Direct Method)
            pred_new = self.agent.prior_model.predict(context, p_new)
            exp_rev_new = p_new * pred_new.prob
            
            # 2. Predict Outcome for HISTORICAL Price (Bias Baseline)
            pred_hist = self.agent.prior_model.predict(context, p_hist)
            exp_rev_hist = p_hist * pred_hist.prob
            
            # --- D. Doubly Robust Correction ---
            # We define a "match" window. If prices are close, we use the real data to correct bias.
            distance = abs(p_hist - p_new)
            bandwidth = 10.0 # $10 window
            
            if distance < bandwidth:
                # We have a match! Use the observed error to correct the model.
                # Correction = Real_Revenue - Predicted_Historical_Revenue
                correction = rev_hist - exp_rev_hist
                dr_value = exp_rev_new + correction
            else:
                # No match. We have to trust the model blindly (Direct Method).
                dr_value = exp_rev_new
                
            dr_values.append(dr_value)
            
        # 3. Aggregation & Metrics
        est_policy_rev = np.sum(dr_values)
        
        # Calculate Uplift
        if hist_revenue > 0:
            uplift_pct = (est_policy_rev - hist_revenue) / hist_revenue
        else:
            uplift_pct = 0.0
        
        # Simple Confidence Interval (Asymptotic Normal)
        std_err = np.std(dr_values) / np.sqrt(n)
        margin = 1.96 * std_err * n # Scale to total sum
        
        return EvalMetric(
            policy_name="TrustAware_Bandit_v1",
            total_revenue_est=est_policy_rev,
            uplift_pct=uplift_pct,
            confidence_interval=(est_policy_rev - margin, est_policy_rev + margin)
        )