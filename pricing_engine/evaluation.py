import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from scipy.stats import ks_2samp

# ============================================================
# 1. Result Containers (The Schema)
# ============================================================

@dataclass
class WeightDiagnostics:
    """
    Health check for Importance Sampling weights.
    If ESS is low, the OPE estimate is unstable (dominated by outliers).
    """
    max_weight: float
    mean_weight: float
    effective_sample_size: float  # ESS
    ess_ratio: float              # ESS / N

@dataclass
class OPEEstimate:
    """Standardized result for Revenue Estimation."""
    estimator_name: str
    expected_reward: float
    ci_lower: float
    ci_upper: float
    std_error: float
    sample_size: int
    diagnostics: Optional[WeightDiagnostics] = None

@dataclass
class CalibrationResult:
    """Standardized result for Model Reality Check."""
    model_name: str
    brier_score: float
    ece: float

@dataclass
class DriftResult:
    """Standardized result for Data Stability."""
    feature: str
    ks_stat: float
    p_value: float
    is_shifted: bool

@dataclass
class AuditReport:
    """
    The Executive Summary.
    Bundles Math, Risk, and Physics checks into a single deployable artifact.
    """
    policy_name: str
    ope_results: Dict[str, OPEEstimate] # e.g. {"DR": ..., "SNIPS": ...}
    risk_metrics: Dict[str, float]      # e.g. {"CVaR_0.05": ...}
    passed: bool = False
    calibration: Optional[CalibrationResult] = None
    drift: List[DriftResult] = field(default_factory=list)

# ============================================================
# 2. Risk & Safety Profiler
# ============================================================

class RiskProfiler:
    """
    Quantifies 'Ruin Probability' (CVaR).
    WARNING: Input rewards must be profit-aligned (Higher = Better).
    """
    @staticmethod
    def calculate_cvar(rewards: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
        """
        Conditional Value at Risk (CVaR).
        "If we hit the worst 5% of scenarios, what is the average revenue?"
        """
        if rewards is None or len(rewards) == 0:
            return {"VaR": 0.0, "CVaR": 0.0}
            
        rewards = np.nan_to_num(rewards, nan=0.0)
        
        # 1. Identify the worst-case threshold (VaR)
        var = np.percentile(rewards, 100 * alpha)
        
        # 2. Average the losses below that threshold
        tail = rewards[rewards <= var]
        cvar = tail.mean() if len(tail) > 0 else var
        
        return {f"VaR_{alpha}": float(var), f"CVaR_{alpha}": float(cvar)}

# ============================================================
# 3. Off-Policy Evaluator (The Auditor)
# ============================================================

class OffPolicyEvaluator:
    """
    The Revenue Auditor. 
    Uses Causal Inference to value policies offline.
    """
    def __init__(self, n_bootstrap: int = 1000, ci_alpha: float = 0.05):
        self.n_bootstrap = n_bootstrap
        self.ci_alpha = ci_alpha

    def _calculate_diagnostics(self, weights: np.ndarray) -> WeightDiagnostics:
        """Computes ESS and weight stats."""
        if len(weights) == 0:
            return WeightDiagnostics(0.0, 0.0, 0.0, 0.0)
            
        sum_w = np.sum(weights)
        sum_sq_w = np.sum(weights ** 2)
        
        # Effective Sample Size (Kish's approximation)
        ess = (sum_w ** 2) / (sum_sq_w + 1e-9)
        
        return WeightDiagnostics(
            max_weight=float(np.max(weights)),
            mean_weight=float(np.mean(weights)),
            effective_sample_size=float(ess),
            ess_ratio=float(ess / len(weights))
        )

    def evaluate_direct_method(self, predicted_rewards: np.ndarray) -> OPEEstimate:
        """Direct Method (DM): Trusts the Model."""
        preds = np.nan_to_num(predicted_rewards)
        mean_val = preds.mean()
        boots = self._bootstrap_mean(preds)

        return OPEEstimate(
            estimator_name="Direct Method (DM)",
            expected_reward=float(mean_val),
            ci_lower=float(np.percentile(boots, 100 * self.ci_alpha / 2)),
            ci_upper=float(np.percentile(boots, 100 * (1 - self.ci_alpha / 2))),
            std_error=float(np.std(boots)),
            sample_size=len(preds),
            diagnostics=None 
        )

    def evaluate_ipw(
        self,
        observed_rewards: np.ndarray,
        logging_propensities: np.ndarray,
        target_propensities: np.ndarray,
        normalize: bool = False
    ) -> OPEEstimate:
        """IPW / SNIPS Estimator. Trusts the Propensities."""
        # Sanitization
        observed_rewards = np.nan_to_num(observed_rewards)
        logging_propensities = np.clip(logging_propensities, 1e-3, 1.0)

        # Weights
        weights = target_propensities / logging_propensities
        
        if normalize:
            # Self-Normalized IPW (SNIPS)
            norm_factor = np.sum(weights) + 1e-9
            weighted_rewards = (weights * observed_rewards) / (norm_factor / len(weights))
            est_name = "SNIPS"
        else:
            # Standard IPW
            weighted_rewards = weights * observed_rewards
            est_name = "IPW"

        mean_val = np.mean(weighted_rewards)
        boots = self._bootstrap_mean(weighted_rewards)
        diag = self._calculate_diagnostics(weights)

        return OPEEstimate(
            estimator_name=est_name,
            expected_reward=float(mean_val),
            ci_lower=float(np.percentile(boots, 100 * self.ci_alpha / 2)),
            ci_upper=float(np.percentile(boots, 100 * (1 - self.ci_alpha / 2))),
            std_error=float(np.std(boots)),
            sample_size=len(observed_rewards),
            diagnostics=diag
        )

    def evaluate_doubly_robust(
        self,
        observed_rewards: np.ndarray,
        logging_propensities: np.ndarray,
        target_propensities: np.ndarray,
        estimated_rewards_obs: np.ndarray,
        estimated_rewards_target: np.ndarray
    ) -> OPEEstimate:
        """Doubly Robust (DR). The Gold Standard."""
        # Sanitization
        obs_r = np.nan_to_num(observed_rewards)
        est_r_obs = np.nan_to_num(estimated_rewards_obs)
        est_r_tgt = np.nan_to_num(estimated_rewards_target)
        log_p = np.clip(logging_propensities, 1e-3, 1.0)

        # Weights
        weights = target_propensities / log_p
        
        # DR Logic
        correction = weights * (obs_r - est_r_obs)
        dr_values = est_r_tgt + correction
        
        mean_val = dr_values.mean()
        boots = self._bootstrap_mean(dr_values)
        diag = self._calculate_diagnostics(weights)

        return OPEEstimate(
            estimator_name="Doubly Robust (DR)",
            expected_reward=float(mean_val),
            ci_lower=float(np.percentile(boots, 100 * self.ci_alpha / 2)),
            ci_upper=float(np.percentile(boots, 100 * (1 - self.ci_alpha / 2))),
            std_error=float(np.std(boots)),
            sample_size=len(obs_r),
            diagnostics=diag
        )

    def _bootstrap_mean(self, values: np.ndarray) -> np.ndarray:
        n = len(values)
        if n == 0: return np.array([0.0])
        indices = np.random.randint(0, n, size=(self.n_bootstrap, n))
        return values[indices].mean(axis=1)

# ============================================================
# 4. Calibration Evaluator & Drift Scanner
# ============================================================

class CalibrationEvaluator:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model") -> CalibrationResult:
        y_true = np.array(y_true)
        y_prob = np.clip(np.array(y_prob), 0.0, 1.0)
        brier = np.mean((y_prob - y_true) ** 2)
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        ece = 0.0
        total_samples = len(y_true)
        for i in range(self.n_bins):
            mask = bin_indices == i
            n_in_bin = np.sum(mask)
            if n_in_bin > 0:
                ece += (n_in_bin / total_samples) * np.abs(np.mean(y_true[mask]) - np.mean(y_prob[mask]))
        return CalibrationResult(model_name=model_name, brier_score=float(brier), ece=float(ece))

class DriftScanner:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def detect_shift(self, ref_df: pd.DataFrame, curr_df: pd.DataFrame, features: List[str]) -> List[DriftResult]:
        results = []
        for f in features:
            if f in ref_df.columns and f in curr_df.columns:
                ref = ref_df[f].dropna().values
                curr = curr_df[f].dropna().values
                if len(ref) > 0 and len(curr) > 0:
                    stat, p_val = ks_2samp(ref, curr)
                    results.append(DriftResult(feature=f, ks_stat=float(stat), p_value=float(p_val), is_shifted=p_val < self.alpha))
        return results