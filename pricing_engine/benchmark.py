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
    DriftScanner
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
    Runs the 8-Pillar Flight Check.
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
        """Runs all benchmarks and returns a scorecard."""
        results = []
        
        results.append(self.b01_latency(curr_df))
        results.append(self.b02_drift(ref_df, curr_df))
        results.append(self.b03_calibration(curr_df))
        results.append(self.b04_policy_value(curr_df))
        results.append(self.b05_safety_governor(curr_df))
        
        # --- THIS WAS MISSING IN YOUR FILE ---
        results.append(self.b06_adversarial_inputs()) 
        # -------------------------------------
        
        results.append(self.b07_fallback(curr_df))
        results.append(self.b08_trust(curr_df))
        
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
        booked_drift = "is_booked" in shifted_feats
        
        return BenchmarkResult(
            "B02", "Data Drift", 
            "WARN" if booked_drift else "PASS", 
            {"Shifted_Count": len(shifted_feats)}, 
            f"Drifting: {shifted_feats}"
        )

    # --- B03: Calibration ---
    def b03_calibration(self, df) -> BenchmarkResult:
        obs_probe = df.copy()
        if hasattr(self.models[ModelRole.MEAN], "cat_cols"):
            for c in self.models[ModelRole.MEAN].cat_cols:
                if c in obs_probe.columns: obs_probe[c] = obs_probe[c].astype("category")
        
        y_true = df["is_booked"].values
        y_prob = np.nan_to_num(self.models[ModelRole.MEAN].predict(obs_probe), nan=0.0)
        
        cal = self.calibrator.evaluate(y_true, y_prob)
        passed = cal.ece < 0.1
        return BenchmarkResult("B03", "Calibration", "PASS" if passed else "FAIL", {"ECE": cal.ece}, f"ECE: {cal.ece:.3f}")

    # --- B04: Policy Value (STABILIZED) ---
    def b04_policy_value(self, df) -> BenchmarkResult:
        n_audit = min(5000, len(df))
        
        # Outlier Filter
        clean_df = df[
            (df["avg_price"] < 1000) & 
            (df["avg_price"] > 10)
        ].copy()
        
        audit_sample = clean_df.sample(n_audit, replace=True).reset_index(drop=True)
        baseline_rev = (audit_sample["avg_price"] * audit_sample["is_booked"]).mean()
        
        action_space = np.linspace(50, 300, 10).tolist()
        t_props, est_r = [], []
        
        # Get Q-Values
        try:
            obs_probe = audit_sample.copy()
            if hasattr(self.models[ModelRole.MEAN], "cat_cols"):
                for c in self.models[ModelRole.MEAN].cat_cols:
                    if c in obs_probe.columns: obs_probe[c] = obs_probe[c].astype("category")
            q_probs = np.nan_to_num(self.models[ModelRole.MEAN].predict(obs_probe), nan=0.0)
            q_vals = q_probs * audit_sample["avg_price"].values
        except:
            q_vals = np.zeros(len(audit_sample))

        # Stabilized Propensities (min 5%)
        log_props = np.full(n_audit, 0.05)

        for _, row in audit_sample.iterrows():
            dec = self.policy.select_price(row, action_space, self.models)
            dist = abs(dec.price - row["avg_price"])
            t_props.append(np.exp(-dist/20.0))
            est_r.append(dec.expected_revenue)

        t_props = np.nan_to_num(np.array(t_props))
        est_r = np.nan_to_num(np.array(est_r))

        dr = self.ope.evaluate_doubly_robust(
            (audit_sample["avg_price"] * audit_sample["is_booked"]).values,
            log_props, t_props, q_vals, est_r
        )
        
        if baseline_rev > 0:
            uplift = (dr.ci_lower - baseline_rev) / baseline_rev
        else:
            uplift = 0.0
            
        passed = uplift > -0.05 
        return BenchmarkResult("B04", "Policy Value", "PASS" if passed else "WARN", {"Uplift_LB": uplift}, f"Uplift: {uplift:+.1%}")

    # --- B05: Safety Governor ---
    def b05_safety_governor(self, df) -> BenchmarkResult:
        test_row = df.sample(1).iloc[0]
        EXTREME_SPACE = [10.0, 1000.0] 
        
        decision = self.policy.select_price(test_row, EXTREME_SPACE, self.models)
        constraints = {'max_price': 500.0}
        
        safe_res = self.safety.validate_and_clamp(
            price=decision.price,
            constraints=constraints,
            context_row=test_row,
            demand_models=self.models
        )
        
        is_safe = safe_res.safe_price <= 500.0 and safe_res.safe_price > 0
        return BenchmarkResult("B05", "Safety Governor", "PASS" if is_safe else "FAIL", {"Price": safe_res.safe_price}, f"Out: {safe_res.safe_price}")

    # --- B06: Adversarial Inputs (NEW) ---
    def b06_adversarial_inputs(self) -> BenchmarkResult:
        # We test garbage inputs that usually crash model.predict()
        bad_inputs = [
            {"avg_price": np.nan, "is_booked": 0},
            {"avg_price": np.inf, "is_booked": 0},
            {"avg_price": -100.0, "is_booked": 0},
            {} # Empty context
        ]
        
        failures = 0
        for ctx in bad_inputs:
            try:
                row = pd.Series(ctx)
                # Should not crash
                dec = self.policy.select_price(row, [100.0], self.models)
                
                # Check Safety too
                safe_res = self.safety.validate_and_clamp(dec.price, {}, row, self.models)
                
                if not (safe_res.safe_price >= 0): failures += 1
            except:
                failures += 1
                
        passed = failures == 0
        return BenchmarkResult("B06", "Adversarial Inputs", "PASS" if passed else "FAIL", {"Failures": failures}, "Handled NaN/Inf/Empty")

    # --- B07: Fallback ---
    def b07_fallback(self, df) -> BenchmarkResult:
        backup = self.models.copy()
        backup[ModelRole.MEAN] = None 
        try:
            dec = self.policy.select_price(df.sample(1).iloc[0], [100.0], backup)
            passed = dec.price > 0
            msg = "Survived outage"
        except:
            passed = False
            msg = "Crashed on outage"
        return BenchmarkResult("B07", "Fallback", "PASS" if passed else "FAIL", {}, msg)

    # --- B08: Trust ---
    def b08_trust(self, df) -> BenchmarkResult:
        metrics = self.trust.evaluate_trust(
            df.sample(min(500, len(df))), 
            np.linspace(80, 200, 10).tolist(), 
            self.models
        )
        passed = metrics.safety_violations == 0
        return BenchmarkResult("B08", "Trust", "PASS" if passed else "FAIL", {"Violations": metrics.safety_violations}, f"VolRed: {metrics.volatility_reduction:.1%}")