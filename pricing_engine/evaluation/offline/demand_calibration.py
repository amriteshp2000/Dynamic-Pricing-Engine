# pricing_engine/evaluation/offline/demand_calibration.py

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    model_name: str
    brier_score: float
    ece: float


class DemandCalibrationEvaluator:
    """
    Evaluates probabilistic correctness of demand models.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def evaluate(self, demand_model, data: pd.DataFrame):
        # Safe model name resolution
        model_name = getattr(
            demand_model,
            "name",
            demand_model.__class__.__name__
        )

        y_true = data["is_booked"].values
        y_prob = data.apply(
            lambda r: demand_model.predict(r.to_dict(), r.price).prob,
            axis=1
        ).values

        # Brier score
        brier = np.mean((y_prob - y_true) ** 2)

        # Expected Calibration Error (ECE)
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += (mask.mean()) * abs(acc - conf)

        return CalibrationResult(
            model_name=model_name,
            brier_score=brier,
            ece=ece
        )
