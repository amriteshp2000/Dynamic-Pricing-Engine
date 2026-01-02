# pricing_engine/evaluation/offline/safety_impact.py

from dataclasses import dataclass


@dataclass
class SafetyImpactResult:
    policy_name: str
    clamp_rate: float
    violation_count: int
    override_rate: float   # behavioral deviation from historical pricing


class SafetyImpactEvaluator:
    """
    Measures safety intervention and behavioral aggressiveness of a policy.
    """

    def __init__(self, override_threshold: float = 0.30):
        """
        Parameters
        ----------
        override_threshold : float
            Relative deviation from historical price beyond which
            a decision is considered an override (e.g., 0.30 = 30%)
        """
        self.override_threshold = override_threshold

    def evaluate(self, policy, history, safety, constraints_fn):
        clamps = 0
        violations = 0
        overrides = 0

        for _, row in history.iterrows():
            ctx = row.to_dict()
            p_hist = row["price"]

            decision = policy.choose_price(ctx)
            res = safety.enforce(
                decision.selected_price,
                constraints_fn(ctx, row)
            )

            p_new = res.safe_price

            # Safety clamp
            if res.was_clamped:
                clamps += 1

            # Safety violation (hard fail)
            if not res.is_valid:
                violations += 1

            # Behavioral override (independent of safety)
            if p_hist > 0:
                deviation = abs(p_new - p_hist) / p_hist
                if deviation > self.override_threshold:
                    overrides += 1

        n = len(history)

        return SafetyImpactResult(
            policy_name=policy.name,
            clamp_rate=clamps / n,
            violation_count=violations,
            override_rate=overrides / n
        )
