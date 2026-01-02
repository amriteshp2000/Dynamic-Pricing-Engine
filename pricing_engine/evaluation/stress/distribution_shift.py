# pricing_engine/evaluation/stress/distribution_shift.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import ks_2samp

@dataclass
class DistributionShiftResult:
    feature: str
    ks_stat: float
    p_value: float
    shifted: bool


class DistributionShiftTester:
    """
    Detects covariate and outcome drift using KS tests.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def test(self, reference: pd.DataFrame, current: pd.DataFrame, features):
        results = []

        for f in features:
            ref_vals = reference[f].dropna()
            cur_vals = current[f].dropna()

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            ks, p = ks_2samp(ref_vals, cur_vals)
            results.append(
                DistributionShiftResult(
                    feature=f,
                    ks_stat=ks,
                    p_value=p,
                    shifted=p < self.alpha
                )
            )

        return results
