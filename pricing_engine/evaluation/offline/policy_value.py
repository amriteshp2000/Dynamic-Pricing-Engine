# pricing_engine/evaluation/offline/policy_value.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PolicyValueResult:
    policy_name: str
    method: str
    est_revenue: float
    uplift_pct: float
    ci_95: Tuple[float, float]


class PolicyValueEvaluator:
    """
    Offline policy value estimation.
    Supports Direct, IPS, and Doubly Robust estimators.
    """

    def __init__(self, safety_governor, bandwidth: float = 10.0):
        self.safety = safety_governor
        self.bandwidth = bandwidth

    def doubly_robust(self, policy, demand_model, history, constraints_fn):
        dr_vals = []
        hist_rev = (history.price * history.is_booked).sum()

        for _, row in history.iterrows():
            ctx = row.to_dict()
            p_hist, y = row.price, row.is_booked
            r_hist = p_hist * y

            decision = policy.choose_price(ctx)
            safe = self.safety.enforce(
                decision.selected_price,
                constraints_fn(ctx, row)
            )
            p_new = safe.safe_price

            pred_new = demand_model.predict(ctx, p_new)
            pred_hist = demand_model.predict(ctx, p_hist)

            exp_new = p_new * pred_new.prob
            exp_hist = p_hist * pred_hist.prob

            if abs(p_new - p_hist) <= self.bandwidth:
                dr_vals.append(exp_new + (r_hist - exp_hist))
            else:
                dr_vals.append(exp_new)

        dr_vals = np.array(dr_vals)
        est = dr_vals.sum()
        uplift = (est - hist_rev) / hist_rev if hist_rev > 0 else 0.0

        se = dr_vals.std(ddof=1) / np.sqrt(len(dr_vals))
        margin = 1.96 * se * len(dr_vals)

        return PolicyValueResult(
            policy_name=policy.name,
            method="DoublyRobust",
            est_revenue=est,
            uplift_pct=uplift,
            ci_95=(est - margin, est + margin)
        )
