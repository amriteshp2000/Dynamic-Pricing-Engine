# pricing_engine/evaluation/stress/cold_start.py

from dataclasses import dataclass

@dataclass
class ColdStartResult:
    policy_name: str
    fallback_used: bool
    valid_price: bool


class ColdStartTester:
    """
    Evaluates policy behavior with empty or near-empty history.
    """

    def test(self, policy, safety, default_constraints):
        try:
            decision = policy.choose_price({})
            safe = safety.enforce(
                decision.selected_price,
                default_constraints
            )

            return ColdStartResult(
                policy_name=policy.name,
                fallback_used=decision.metadata.get("fallback", False),
                valid_price=safe.safe_price is not None
            )
        except Exception:
            return ColdStartResult(
                policy_name=policy.name,
                fallback_used=False,
                valid_price=False
            )
