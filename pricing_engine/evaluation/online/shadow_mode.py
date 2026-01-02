# pricing_engine/evaluation/online/shadow_mode.py

class ShadowPolicyRunner:
    """
    Runs new policy in parallel without affecting prices.
    """

    def __init__(self, shadow_policy):
        self.policy = shadow_policy
        self.decisions = []

    def observe(self, context):
        decision = self.policy.choose_price(context)
        self.decisions.append(decision)
