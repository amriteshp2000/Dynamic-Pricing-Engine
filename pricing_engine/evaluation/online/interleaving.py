# pricing_engine/evaluation/online/interleaving.py

import random

class InterleavingEvaluator:
    """
    Alternates between control and treatment safely.
    """

    def choose(self, control_price, treatment_price):
        return control_price if random.random() < 0.5 else treatment_price
