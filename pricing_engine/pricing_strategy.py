import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# IMPORT THE ROBUST GOVERNOR
from pricing_engine.safety import SafetyGovernor, SafetyConfig

# ============================================================
# 1. Configuration & Contracts
# ============================================================

@dataclass
class BanditDecision:
    price: float
    prob_book: float
    expected_revenue: float
    scores: Dict[float, float]
    policy_name: str
    is_exploratory: bool
    context_id: Any = None
    uncertainty_proxy: float = 0.0
    
    was_overridden: bool = False
    override_reason: Optional[str] = None
    meta_info: Dict[str, Any] = field(default_factory=dict)

class ModelRole:
    MEAN = "LGBM_Tweedie"
    UNCERTAINTY = "HierarchicalBayes"
    SAFETY = "TF_Lattice"
    RESEARCH = "DeepFM"

# ============================================================
# 2. Abstract Base Class
# ============================================================

class PricingPolicy(ABC):
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    @abstractmethod
    def select_price(
        self, 
        context_row: pd.Series, 
        valid_prices: List[float],
        demand_models: Dict[str, Any]
    ) -> BanditDecision:
        pass

    def _make_probe(self, row, prices, model) -> pd.DataFrame:
        df = pd.DataFrame([row] * len(prices))
        df["avg_price"] = prices
        df["log_price"] = np.log1p(prices)
        if hasattr(model, "cat_cols"):
             for c in model.cat_cols:
                 if c in df.columns: df[c] = df[c].astype("category")
        return df

# ============================================================
# 3. Strategy Adapter for Safety Governor
# ============================================================

class StrategyGovernorAdapter:
    """
    Wraps the robust SafetyGovernor to work with BanditDecisions.
    """
    def __init__(self, config: SafetyConfig = SafetyConfig()):
        self.core_governor = SafetyGovernor(config)

    def govern(self, decision: BanditDecision, demand_models: Dict[str, Any], context_row: pd.Series) -> BanditDecision:
        
        # 1. Call the robust safety engine
        # We pass empty constraints dict for simulation, but in prod this would come from DB
        result = self.core_governor.validate_and_clamp(
            suggested_price=decision.price,
            predicted_prob=decision.prob_book,
            context_row=context_row,
            demand_models=demand_models,
            constraints={"min_price": 75.0}, # Simulation floor
            prev_price=None, # Simulation is stateless for now
            demand_meta={"std_dev": decision.uncertainty_proxy}
        )
        
        if not result.is_clamped:
            return decision

        # 2. If clamped, we must re-calculate expectations for consistency
        mean_model = demand_models.get(ModelRole.MEAN)
        new_prob = decision.prob_book
        
        if mean_model:
            probe = pd.DataFrame([context_row])
            probe["avg_price"] = result.safe_price
            probe["log_price"] = np.log1p(result.safe_price)
            if hasattr(mean_model, "cat_cols"):
                 for c in mean_model.cat_cols:
                     if c in probe.columns: probe[c] = probe[c].astype("category")
            try:
                new_prob = mean_model.predict(probe)[0]
            except:
                pass

        return BanditDecision(
            price=result.safe_price,
            prob_book=new_prob,
            expected_revenue=result.safe_price * new_prob,
            scores=decision.scores,
            policy_name=decision.policy_name,
            is_exploratory=False,
            context_id=decision.context_id,
            uncertainty_proxy=decision.uncertainty_proxy,
            was_overridden=True,
            override_reason=result.trigger_reason,
            meta_info=decision.meta_info
        )

# ============================================================
# 4. The 5-Bandit Portfolio (Unchanged Logic, just Context)
# ============================================================

# [Include the same 5 bandit classes here: ThompsonSamplingPolicy, BayesianUCBPolicy, etc.]
# ... (Paste the bandit classes from previous response exactly as is) ...

# --- B1: Thompson Sampling (Bayesian Standard) ---
class ThompsonSamplingPolicy(PricingPolicy):
    def __init__(self, model_key: str = ModelRole.UNCERTAINTY):
        super().__init__(name="ThompsonSampling", role="deployable")
        self.model_key = model_key

    def select_price(self, context_row, valid_prices, demand_models):
        model = demand_models.get(self.model_key)
        probe_df = self._make_probe(context_row, valid_prices, model)
        mu_preds = model.predict(probe_df)
        sigma = np.maximum(0.05, 0.1 * mu_preds)
        sampled_probs = np.clip(np.random.normal(mu_preds, sigma), 0, 1)
        revenues = np.array(valid_prices) * sampled_probs
        best_idx = np.argmax(revenues)

        return BanditDecision(
            price=valid_prices[best_idx], prob_book=mu_preds[best_idx], 
            expected_revenue=revenues[best_idx], scores=dict(zip(valid_prices, revenues)), 
            policy_name=self.name, is_exploratory=True, uncertainty_proxy=sigma[best_idx]
        )

# --- B2: Bayesian UCB ---
class BayesianUCBPolicy(PricingPolicy):
    def __init__(self, model_key=ModelRole.UNCERTAINTY, ucb_scale=1.0):
        super().__init__(name="BayesianUCB", role="deployable")
        self.model_key = model_key
        self.alpha = ucb_scale

    def select_price(self, context_row, valid_prices, demand_models):
        model = demand_models[self.model_key]
        probe_df = self._make_probe(context_row, valid_prices, model)
        mu = model.predict(probe_df)
        sigma = 0.1 * mu 
        optimistic_probs = np.clip(mu + (self.alpha * sigma), 0, 1)
        revenues = np.array(valid_prices) * optimistic_probs
        best_idx = np.argmax(revenues)
        return BanditDecision(
            price=valid_prices[best_idx], prob_book=mu[best_idx], 
            expected_revenue=revenues[best_idx], scores=dict(zip(valid_prices, revenues)), 
            policy_name=self.name, is_exploratory=False, uncertainty_proxy=sigma[best_idx]
        )

# --- B3: Model Uncertainty TS ---
class ModelUncertaintyTSPolicy(PricingPolicy):
    def __init__(self, candidates=[ModelRole.MEAN, ModelRole.RESEARCH, ModelRole.UNCERTAINTY]):
        super().__init__(name="ModelUncertaintyTS", role="deployable")
        self.candidates = candidates

    def select_price(self, context_row, valid_prices, demand_models):
        available = [m for m in self.candidates if m in demand_models]
        chosen_key = np.random.choice(available)
        model = demand_models[chosen_key]
        probe_df = self._make_probe(context_row, valid_prices, model)
        preds = model.predict(probe_df)
        noise = np.random.normal(0, 0.05 * preds)
        perturbed_preds = np.clip(preds + noise, 0, 1)
        revenues = np.array(valid_prices) * perturbed_preds
        best_idx = np.argmax(revenues)
        return BanditDecision(
            price=valid_prices[best_idx], prob_book=preds[best_idx], 
            expected_revenue=revenues[best_idx], scores=dict(zip(valid_prices, revenues)), 
            policy_name=self.name, is_exploratory=True, uncertainty_proxy=0.05 * preds[best_idx],
            meta_info={"trusted_model": chosen_key}
        )

# --- B4: Greedy Confidence ---
class GreedyConfidencePolicy(PricingPolicy):
    def __init__(self, mean_model=ModelRole.MEAN, unc_model=ModelRole.UNCERTAINTY, penalty_scale=0.5):
        super().__init__(name="GreedyConfidence", role="deployable")
        self.mean_key = mean_model
        self.unc_key = unc_model
        self.lam = penalty_scale

    def select_price(self, context_row, valid_prices, demand_models):
        mu_model = demand_models[self.mean_key]
        probe_mu = self._make_probe(context_row, valid_prices, mu_model)
        mu_preds = mu_model.predict(probe_mu)
        sigma_model = demand_models[self.unc_key]
        probe_sigma = self._make_probe(context_row, valid_prices, sigma_model)
        sigma_preds = 0.1 * sigma_model.predict(probe_sigma)
        exp_rev = np.array(valid_prices) * mu_preds
        rev_risk = np.array(valid_prices) * sigma_preds
        scores = exp_rev - (self.lam * rev_risk)
        best_idx = np.argmax(scores)
        return BanditDecision(
            price=valid_prices[best_idx], prob_book=mu_preds[best_idx], 
            expected_revenue=exp_rev[best_idx], scores=dict(zip(valid_prices, scores)), 
            policy_name=self.name, is_exploratory=False, uncertainty_proxy=sigma_preds[best_idx],
            meta_info={"revenue_component": exp_rev[best_idx], "risk_penalty": rev_risk[best_idx]}
        )

# --- B5: Epsilon Greedy ---
class EpsilonGreedyPolicy(PricingPolicy):
    def __init__(self, model_key=ModelRole.MEAN, epsilon=0.1):
        super().__init__(name="EpsilonGreedy", role="baseline")
        self.model_key = model_key
        self.epsilon = epsilon

    def select_price(self, context_row, valid_prices, demand_models):
        model = demand_models[self.model_key]
        probe_df = self._make_probe(context_row, valid_prices, model)
        mu = model.predict(probe_df)
        revenues = np.array(valid_prices) * mu
        if np.random.rand() < self.epsilon:
            best_idx = np.random.randint(len(valid_prices))
            is_exploratory = True
        else:
            best_idx = np.argmax(revenues)
            is_exploratory = False
        return BanditDecision(
            price=valid_prices[best_idx], prob_book=mu[best_idx], 
            expected_revenue=revenues[best_idx], scores=dict(zip(valid_prices, revenues)), 
            policy_name=self.name, is_exploratory=is_exploratory, uncertainty_proxy=0.0
        )