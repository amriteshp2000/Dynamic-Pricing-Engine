# pricing_engine/__init__.py

# ---------------------------------------------------------------------
# Demand & Causal Models
# ---------------------------------------------------------------------

from .causal_model import FixedEffectElasticity, LinearDMLElasticity, CausalForestElasticity, run_causal_pipeline

from .demand_model import (
    DemandModel,
    LGBMTweedie,
    DeepFMModel,
    TFLatticeModel,
    HierarchicalBayesianLogit,
    select_model,
)


# ---------------------------------------------------------------------
# Pricing Bandits / Policies
# ---------------------------------------------------------------------

from .Bandit import (
    BasePricingBandit,
    ThompsonPricingBandit,
    BayesianUCBBandit,
    LinUCBBandit,
    StreamingBayesianLogistic,
    PricingDecision,
    SafetyGatedBandit,
    EnterpriseSafeBandit,
)

# ---------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------

from .safety import SafetyGovernor, SafetyConfig

# ---------------------------------------------------------------------
# Evaluation Suite (EXPOSED AS MODULE)
# ---------------------------------------------------------------------

from . import evaluation

# ---------------------------------------------------------------------
# Data Utilities
# ---------------------------------------------------------------------

from .data_loader import load_and_clean_seattle_data

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    # Demand & Causal
    "DemandModel",
    "LGBMTweedie",
    "DeepFMModel",
    "TFLatticeModel",
    "HierarchicalBayesianLogit",
    "select_model",


    # Bandits
    "BasePricingBandit",
    "ThompsonPricingBandit",
    "BayesianUCBBandit",
    "LinUCBBandit",
    "StreamingBayesianLogistic",
    "PricingDecision",
    "SafetyGatedBandit",
    "EnterpriseSafeBandit",
    

    # Safety
    "SafetyGovernor", "SafetyConfig",

    # Evaluation (module-level)
    "evaluation",

    # Data
    "load_and_clean_seattle_data",
]
