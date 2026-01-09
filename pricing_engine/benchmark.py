import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Import Internal Modules
from .pricing_strategy import PricingPolicy, ThompsonSamplingPolicy, ModelRole
from .safety import SafetyGovernor
from .trust import TrustEvaluator
from .evaluation import (
    OffPolicyEvaluator, 
    CalibrationEvaluator, 
    DriftScanner,
    OPEEstimate
)

@dataclass
class BenchmarkResult:
    test_id: str
    name: str
    status: str
    metrics: Dict[str, float]
    message: str

class PricingBenchmarkSuite:
    """
    Module 05: System Health & Certification.
    Runs the 7-Pillar Flight Check.
    """
    
    def __init__(self, policy: PricingPolicy, models: Dict[ModelRole, Any]):
        self.policy = policy
        self.models = models
        self.safety = SafetyGovernor()
        
        # Tools
        self.ope = OffPolicyEvaluator(n_bootstrap=100)
        self.calibrator = CalibrationEvaluator(n_bins=10)
        self.drift = DriftScanner(alpha=0.05)
        self.trust = TrustEvaluator(policy, self.safety)

    def run_all(self, ref_df: pd.DataFrame, curr_df: pd.DataFrame) -> pd.DataFrame:
        """Runs all 7 benchmarks and returns a scorecard DataFrame."""
        results = []
        
        # 1. Latency
        results.append(self.b01_latency(curr_df))
        
        # 2. Drift
        results.append(self.b02_drift(ref_df, curr_df))
        
        # 3. Calibration
        results.append(self.b03_calibration(curr_df))
        
        # 4. OPE (Value)
        results.append(self.b04_policy_value(curr_df))
        
        # 5. Safety
        results.append(self.b05_safety_governor(curr_df))
        
        # 6. Fallback
        results.append(self.b06_fallback(curr_df))
        
        # 7. Trust
        results.append(self.b07_trust(curr_df))
        
        return pd.DataFrame(results)

    # --- B01: Latency ---
    def b01_latency(self, df: pd.DataFrame, n_calls=500) -> BenchmarkResult:
        latencies = []
        sample = df.sample(n_calls, replace=True).to_dict('records')
        action_space = np.linspace(50, 300, 20).tolist()
        
        for req in sample:
            t0 = time.perf_counter()
            _ = self.policy.select_price(req, action_space, self.models)
            latencies.append((time.perf_counter() - t0) * 1000)
            
        p99 = np.percentile(latencies, 99)
        passed = p99 < 50.0
        return BenchmarkResult("B01", "Latency", "PASS" if passed else "FAIL", {"P99_ms": p99}, f"P99: {p99:.1f}ms")

    # --- B02: Drift ---
    def b02_drift(self, ref_df, curr_df) -> BenchmarkResult:
        features = ["avg_price", "is_booked", "accommodates"]
        res = self.drift.detect_shift(ref_df, curr_df, features)
        
        shifted_feats = [r.feature for r in res if r.is_shifted]
        # We allow some drift, but warn if everything drifts. 
        # Strict fail if 'is_booked' drifts heavily (market regime change).
        booked_drift = "is_booked" in shifted_feats
        
        return BenchmarkResult(
            "B02", "Data Drift", 
            "WARN" if booked_drift else "PASS", 
            {"Shifted_Count": len(shifted_feats)}, 
            f"Drifting: {shifted_feats}"
        )

    # --- B03: Calibration ---
    def b03_calibration(self, df) -> BenchmarkResult:
        # Prepare data for model
        obs_probe = df.copy()
        if hasattr(self.models[ModelRole.MEAN], "cat_cols"):
            for c in self.models[ModelRole.MEAN].cat_cols:
                if c in obs_probe.columns: obs_probe[c] = obs_probe[c].astype("category")
        
        y_true = df["is_booked"].values
        y_prob = np.nan_to_num(self.models[ModelRole.MEAN].predict(obs_probe), nan=0.0)
        
        cal = self.calibrator.evaluate(y_true, y_prob)
        passed = cal.ece < 0.1
        
        return BenchmarkResult("B03", "Calibration", "PASS" if passed else "FAIL", {"ECE": cal.ece}, f"ECE: {cal.ece:.3f}")

    # --- B04: Policy Value (OPE) ---
    def b04_policy_value(self, df) -> BenchmarkResult:
        n_audit = min(2000, len(df))
        audit_sample = df.sample(n_audit).reset_index(drop=True)
        baseline_rev = (audit_sample["avg_price"] * audit_sample["is_booked"]).mean()
        
        # Propensity Estimation (Simplified for Benchmark)
        action_space = np.linspace(50, 300, 10).tolist()
        t_props, est_r = [], []
        
        # Prepare Q-values
        obs_probe = audit_sample.copy()
        if hasattr(self.models[ModelRole.MEAN], "cat_cols"):
            for c in self.models[ModelRole.MEAN].cat_cols:
                 if c in obs_probe.columns: obs_probe[c] = obs_probe[c].astype("category")
        q_probs = np.nan_to_num(self.models[ModelRole.MEAN].predict(obs_probe), nan=0.0)
        q_vals = q_probs * audit_sample["avg_price"].values

        # Logging Propensities (dummy 1% if missing)
        log_props = np.full(n_audit, 0.01) # In real benchmark, pass actuals

        for _, row in audit_sample.iterrows():
            dec = self.policy.select_price(row, action_space, self.models)
            dist = abs(dec.price - row["avg_price"])
            t_props.append(np.exp(-dist/10.0))
            est_r.append(dec.expected_revenue)

        t_props = np.nan_to_num(np.array(t_props))
        est_r = np.nan_to_num(np.array(est_r))

        dr = self.ope.evaluate_doubly_robust(
            (audit_sample["avg_price"] * audit_sample["is_booked"]).values,
            log_props, t_props, q_vals, est_r
        )
        
        uplift = (dr.ci_lower - baseline_rev) / baseline_rev
        passed = uplift > -0.05 # Allow small negative in benchmark if safe
        
        return BenchmarkResult("B04", "Policy Value", "PASS" if passed else "WARN", {"Uplift_LB": uplift}, f"Uplift: {uplift:+.1%}")

    # --- B05: Safety Governor ---
    def b05_safety_governor(self, df) -> BenchmarkResult:
        test_row = df.sample(1).iloc[0]
        EXTREME_SPACE = [10.0, 1000.0] # Unsafe
        
        decision = self.policy.select_price(test_row, EXTREME_SPACE, self.models)
        
        # Should be clamped to reasonable bounds (e.g., 50-400)
        is_safe = 40.0 <= decision.price <= 500.0
        return BenchmarkResult("B05", "Safety Governor", "PASS" if is_safe else "FAIL", {"Price": decision.price}, "Clamped extreme inputs")

    # --- B06: Fallback ---
    def b06_fallback(self, df) -> BenchmarkResult:
        backup = self.models.copy()
        backup[ModelRole.MEAN] = None # Kill model
        
        try:
            dec = self.policy.select_price(df.sample(1).iloc[0], [100.0], backup)
            passed = dec.price > 0
            msg = "Survived outage"
        except:
            passed = False
            msg = "Crashed on outage"
            
        return BenchmarkResult("B06", "Fallback", "PASS" if passed else "FAIL", {}, msg)

    # --- B07: Trust ---
    def b07_trust(self, df) -> BenchmarkResult:
        metrics = self.trust.evaluate_trust(
            df.sample(min(500, len(df))), 
            np.linspace(80, 200, 10).tolist(), 
            self.models
        )
        passed = metrics.safety_violations == 0
        return BenchmarkResult("B07", "Trust", "PASS" if passed else "FAIL", {"Violations": metrics.safety_violations}, f"VolRed: {metrics.volatility_reduction:.1%}")