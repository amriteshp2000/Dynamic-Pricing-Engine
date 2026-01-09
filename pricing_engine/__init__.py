# pricing_engine/__init__.py

# ---------------------------------------------------------------------
# 1. Data Utilities (Module 00)
# ---------------------------------------------------------------------
from .data_loader import load_and_clean_seattle_data

# ---------------------------------------------------------------------
# 2. Causal Models (Module 01)
# ---------------------------------------------------------------------
# Keeping these active as they provide the elasticity priors
from .causal_model import (
    FixedEffectElasticity, 
    LinearDMLElasticity, 
    CausalForestElasticity, 
    run_causal_pipeline
)

# ---------------------------------------------------------------------
# 3. Demand Models (Module 02)
# ---------------------------------------------------------------------
from .demand_model import (
    DemandModel,
    LGBMTweedie,
    DeepFMModel,
    TFLatticeModel,
    HierarchicalBayesianLogit,
)

# ---------------------------------------------------------------------
# 4. Pricing Policies / Bandits (Module 03)
# ---------------------------------------------------------------------
# UPDATED: Now exposing the "Policy" architecture
from .pricing_strategy import (
    PricingPolicy,
    ThompsonSamplingPolicy,
    BayesianUCBPolicy,
    ModelUncertaintyTSPolicy,
    GreedyConfidencePolicy,
    EpsilonGreedyPolicy,
    BanditDecision,
    ModelRole,
    StrategyGovernorAdapter
)

# ---------------------------------------------------------------------
# 5. Safety Logic (Module 03 Guardrails)
# ---------------------------------------------------------------------
from .safety import (
    SafetyGovernor, 
    SafetyConfig,
    SafetyResult
)

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    # Data
    "load_and_clean_seattle_data",

    # Causal
    "FixedEffectElasticity",
    "LinearDMLElasticity",
    "CausalForestElasticity",
    "run_causal_pipeline",

    # Demand
    "DemandModel",
    "LGBMTweedie",
    "DeepFMModel",
    "TFLatticeModel",
    "HierarchicalBayesianLogit",

    # Pricing Policies (The 5-Bandit Stack)
    "PricingPolicy",
    "ThompsonSamplingPolicy",
    "BayesianUCBPolicy",
    "ModelUncertaintyTSPolicy",
    "GreedyConfidencePolicy",
    "EpsilonGreedyPolicy",
    
    # Pricing Structs & Adapters
    "BanditDecision",
    "ModelRole",
    "StrategyGovernorAdapter",

    # Safety
    "SafetyGovernor",
    "SafetyConfig",
    "SafetyResult",
]