# pricing_engine/evaluation/offline/attribution.py

from dataclasses import dataclass

@dataclass
class AttributionResult:
    policy_gain: float
    model_gain: float


class AttributionEvaluator:
    """
    Decomposes uplift into:
    - Better pricing decisions
    - Better demand estimation
    """

    def evaluate(self, baseline_policy, new_policy, demand_model, history):
        base_rev = 0.0
        new_rev = 0.0

        for _, row in history.iterrows():
            ctx = row.to_dict()
            p_base = baseline_policy.choose_price(ctx).selected_price
            p_new = new_policy.choose_price(ctx).selected_price

            prob = demand_model.predict(ctx, p_base).prob
            base_rev += p_base * prob
            new_rev += p_new * prob

        return AttributionResult(
            policy_gain=new_rev - base_rev,
            model_gain=0.0  # filled when comparing models
        )
