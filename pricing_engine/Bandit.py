
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging
from .demand_model import HierarchicalDemandModel

logger = logging.getLogger("PriceEngine_Agent")

@dataclass
class PricingDecision:
    selected_price: float
    expected_revenue: float
    uncertainty_sigma: float
    source: str
    panic_mode: bool

class StreamingBayesianRidge:
    """
    Online Bayesian Linear Regression with Forgetting.
    """
    def __init__(self, n_features: int, lambda_forget: float = 0.90, alpha: float = 0.01):
        self.n_features = n_features
        self.lam = lambda_forget
        self.alpha = alpha 
        self.A = self.alpha * np.eye(n_features)
        self.b = np.zeros(n_features)
        self.Sigma = np.linalg.inv(self.A)
        self.mu = np.zeros(n_features)
        self.is_dirty = False
        self.n_updates = 0

    def update(self, x: np.ndarray, y: float):
        x = x.reshape(-1)
        self.A = self.lam * self.A
        self.b = self.lam * self.b
        self.A += np.outer(x, x)
        self.b += y * x
        self.is_dirty = True
        self.n_updates += 1

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        if self.is_dirty:
            self.Sigma = np.linalg.pinv(self.A)
            self.mu = self.Sigma @ self.b
            self.is_dirty = False
        x = x.reshape(-1)
        mean = np.dot(x, self.mu)
        var_pred = np.dot(x, np.dot(self.Sigma, x)) + 0.01 
        return mean, np.sqrt(var_pred)

class ThompsonBandit:
    """
    Module 03: The Agent (With Panic Logic).
    """
    
    def __init__(self, prior_model: HierarchicalDemandModel, forgetting_factor: float = 0.90):
        self.prior_model = prior_model
        self.lam = forgetting_factor
        self.online_models: Dict[str, StreamingBayesianRidge] = {}
        self.consecutive_failures: Dict[str, int] = {} # Track zero-booking streaks

    def _get_feature_vec(self, context: dict, price: float) -> np.ndarray:
        x_raw = np.array([[context[f] for f in self.prior_model.feature_names]])
        x_scaled = self.prior_model.scaler.transform(x_raw)
        log_price = np.log1p(price)
        return np.hstack([x_scaled, [[log_price]]]).flatten()

    def update_belief(self, context: dict, price: float, booked: int):
        lid = str(context['listing_id'])
        x_vec = self._get_feature_vec(context, price)
        
        # Initialize
        if lid not in self.online_models:
            self.n_features = len(x_vec)
            self.online_models[lid] = StreamingBayesianRidge(self.n_features, lambda_forget=self.lam, alpha=0.01)
        
        # Update Model
        self.online_models[lid].update(x_vec, float(booked))
        
        # Update Failure Tracker (Panic Logic)
        if lid not in self.consecutive_failures:
            self.consecutive_failures[lid] = 0
            
        if booked == 0:
            self.consecutive_failures[lid] += 1
        else:
            self.consecutive_failures[lid] = 0 # Reset on success

    def choose_price(self, context: dict, min_p=50, max_p=300) -> PricingDecision:
        lid = str(context['listing_id'])
        
        # --- PANIC LOGIC ---
        # If we failed 3 times in a row, CAP the max price.
        # This forces the agent to explore the 'cheap' region.
        failures = self.consecutive_failures.get(lid, 0)
        is_panic = False
        effective_max = max_p
        
        if failures >= 3:
            is_panic = True
            # Decay max price by 10% for every failure above 2
            # e.g., 3 fails -> max=90%, 5 fails -> max=70%
            decay_factor = max(0.5, 0.9 ** (failures - 2))
            effective_max = max(min_p, max_p * decay_factor)
            
        candidates = np.linspace(min_p, effective_max, 40)
        
        # ... (Standard Fusion Logic) ...
        # Check Trust Switch
        ignore_prior = False
        if lid in self.online_models:
            if self.online_models[lid].n_updates > 5:
                ignore_prior = True
        
        fused_means = []
        fused_stds = []
        
        for p in candidates:
            x_vec = self._get_feature_vec(context, p)
            
            # Online
            m_online, s_online = 0, 0
            has_online = False
            if lid in self.online_models:
                m_online, s_online = self.online_models[lid].predict(x_vec)
                has_online = True
            
            # Prior
            if not ignore_prior:
                hood = context.get('neighborhood')
                if hood in self.prior_model.neighborhood_models:
                    m_prior, s_prior = self.prior_model.neighborhood_models[hood].predict(x_vec.reshape(1,-1), return_std=True)
                else:
                    m_prior, s_prior = self.prior_model.city_model.predict(x_vec.reshape(1,-1), return_std=True)
                m_prior, s_prior = m_prior[0], s_prior[0]
            
            # Fusion
            if ignore_prior and has_online:
                final_mean, final_std = m_online, s_online
            elif has_online:
                w_prior = 1.0 / (s_prior**2 + 1e-6)
                w_online = 1.0 / (s_online**2 + 1e-6)
                final_mean = (m_prior * w_prior + m_online * w_online) / (w_prior + w_online)
                final_std = np.sqrt(1.0 / (w_prior + w_online))
            else:
                final_mean, final_std = m_prior, s_prior
            
            fused_means.append(final_mean)
            fused_stds.append(final_std)
            
        # Thompson Sampling
        fused_means = np.array(fused_means)
        fused_stds = np.array(fused_stds)
        samples = np.random.normal(fused_means, fused_stds)
        samples = np.clip(samples, 0.0, 1.0)
        
        exp_revenues = candidates * samples
        best_idx = np.argmax(exp_revenues)
        
        return PricingDecision(
            selected_price=candidates[best_idx],
            expected_revenue=candidates[best_idx] * fused_means[best_idx],
            uncertainty_sigma=fused_stds[best_idx],
            source="PanicMode" if is_panic else ("Online" if ignore_prior else "Fused"),
            panic_mode=is_panic
        )