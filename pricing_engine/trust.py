import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any
from .pricing_strategy import PricingPolicy, ModelRole
from .safety import SafetyGovernor

@dataclass
class TrustMetrics:
    total_nights: int
    hist_revenue: float
    dr_revenue: float
    uplift_pct: float
    
    # Behavioral Signals
    override_rate: float        # % of prices differing >30% from history
    safety_clamp_rate: float    # % of prices modified by Safety Governor
    safety_violations: int      # Critical failures
    volatility_reduction: float # Is the agent smoother than the human?
def _prepare_model_frame(row: pd.Series, model) -> pd.DataFrame:
    """
    Enforces correct schema for demand model inference.
    """
    df = row.to_frame().T.copy()

    # Cast categoricals
    if hasattr(model, "cat_cols"):
        for c in model.cat_cols:
            if c in df.columns:
                df[c] = df[c].astype("category")

    # Force numeric columns
    for c in df.columns:
        if c not in getattr(model, "cat_cols", []):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df

class TrustEvaluator:
    """
    Evaluates the 'Personality' of the policy.
    Does it behave rationally? Is it stable? Does it fight the human?
    """
    
    def __init__(self, agent: PricingPolicy, safety: SafetyGovernor):
        self.agent = agent
        self.safety = safety

    def evaluate_trust(
        self, 
        df_test: pd.DataFrame, 
        action_space: List[float], 
        models: Dict[ModelRole, Any]
    ) -> TrustMetrics:
        
        n = len(df_test)
        if n == 0: return TrustMetrics(0, 0., 0., 0., 0., 0., 0, 0.)
        
        # Storage
        dr_revenues, hist_revenues = [], []
        agent_prices, hist_prices = [], []
        overrides, clamps, violations = 0, 0, 0
        
        # Loop for granular simulation
        for idx, row in df_test.iterrows():
            p_hist = row['avg_price']
            y_hist = row['is_booked']
            
            # 1. Infer Constraints (Simulation)
            constraints = {
                'min_price': max(10, p_hist * 0.5), 
                'max_price': p_hist * 3.0, 
                'cost_basis': p_hist * 0.4
            }
            
            # 2. Agent Decision
            decision = self.agent.select_price(row, action_space, models)
            
            # 3. Safety Check
            # --- CRITICAL FIX: Use positional arguments to avoid keyword mismatch ---
            # Call signature: validate_and_clamp(price, constraints, context_row, demand_models)
            safe_res = self.safety.validate_and_clamp(
                decision.price,  # Positional arg 1: price
                constraints,     # Positional arg 2: constraints
                row,             # Positional arg 3: context_row
                models           # Positional arg 4: demand_models
            )
            p_agent = safe_res.safe_price
            
            # 4. Behavioral Metrics
            if p_hist > 0:
                if abs(p_agent - p_hist) / p_hist > 0.30: overrides += 1
            
            if safe_res.is_clamped: clamps += 1
            if not safe_res.is_valid: violations += 1
            
            agent_prices.append(p_agent)
            hist_prices.append(p_hist)
            
            # 5. Fast DR Calculation
            ctx = _prepare_model_frame(row, models[ModelRole.MEAN])
            prob_agent = models[ModelRole.MEAN].predict(ctx)[0]

            # Predict
            prob_agent = models[ModelRole.MEAN].predict(ctx)[0]
            prob_hist = models[ModelRole.MEAN].predict(ctx)[0] 
            
            # Doubly Robust Formula
            rev_agent_exp = p_agent * prob_agent
            rev_hist_exp = p_hist * prob_hist
            
            if abs(p_hist - p_agent) < 15.0: 
                dr_val = rev_agent_exp + (p_hist * y_hist - rev_hist_exp)
            else:
                dr_val = rev_agent_exp
                
            dr_revenues.append(dr_val)
            hist_revenues.append(p_hist * y_hist)

        # Aggregation
        total_hist = sum(hist_revenues)
        total_dr = sum(dr_revenues)
        uplift = (total_dr - total_hist) / total_hist if total_hist > 0 else 0.0
        
        vol_hist = np.std(hist_prices)
        vol_agent = np.std(agent_prices)
        vol_reduction = (vol_hist - vol_agent) / vol_hist if vol_hist > 0 else 0.0
        
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