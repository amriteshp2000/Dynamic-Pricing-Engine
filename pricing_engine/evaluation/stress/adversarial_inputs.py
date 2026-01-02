# pricing_engine/evaluation/stress/adversarial_inputs.py

import numpy as np

class AdversarialInputTester:
    """
    Feeds adversarial contexts into the pricing engine.
    """

    def generate_cases(self):
        return [
            {"price": np.nan},
            {"price": np.inf},
            {"price": -1000},
            {"price": 0},
            {"price": 1e6},
            {},  # missing context
        ]

    def test(self, policy, safety, constraints_fn):
        failures = []

        for i, ctx in enumerate(self.generate_cases()):
            try:
                decision = policy.choose_price(ctx)
                safe = safety.enforce(
                    decision.selected_price,
                    constraints_fn(ctx, None)
                )
                if not np.isfinite(safe.safe_price):
                    failures.append((i, "Non-finite price"))
            except Exception as e:
                failures.append((i, str(e)))

        return failures
