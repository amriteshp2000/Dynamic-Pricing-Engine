
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, Tuple
import logging

logger = logging.getLogger("PriceEngine_Demand")

@dataclass
class DemandPrediction:
    prob: float
    std_dev: float
    source_level: str

class HierarchicalDemandModel:
    """
    Module 02: Hierarchical Bayesian Demand Model (Fused).
    
    Implements 'Inverse Variance Weighting' to blend priors and specifics.
    Ensures uncertainty never explodes when switching from Neighborhood -> Listing.
    """
    
    def __init__(self, min_obs_for_listing: int = 10):
        self.min_obs = min_obs_for_listing
        self.scaler = StandardScaler()
        
        self.city_model = BayesianRidge()
        self.neighborhood_models: Dict[str, BayesianRidge] = {}
        self.listing_models: Dict[str, BayesianRidge] = {}
        self.feature_names = []

    def fit(self, df: pd.DataFrame, feature_cols: list):
        logger.info("Training Hierarchical System...")
        self.feature_names = feature_cols
        
        # 1. Prepare Data
        X = df[feature_cols].values
        log_prices = np.log1p(df['price'].values).reshape(-1, 1)
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Full Matrix: [Features | LogPrice]
        train_matrix = np.hstack([X_scaled, log_prices])
        y = df['is_booked'].values
        
        # 2. Train City Model (Level 0)
        self.city_model.fit(train_matrix, y)
        
        # 3. Train Neighborhood Models (Level 1)
        neighborhoods = df['neighborhood'].unique()
        for hood in neighborhoods:
            mask = (df['neighborhood'] == hood)
            if mask.sum() > 10:
                m = BayesianRidge()
                m.fit(train_matrix[mask], y[mask])
                self.neighborhood_models[hood] = m
                
        # 4. Train Listing Models (Level 2) - Pure Data (Fusion handles the prior)
        listing_counts = df['listing_id'].value_counts()
        valid_listings = listing_counts[listing_counts >= self.min_obs].index
        
        logger.info(f"Training {len(valid_listings)} Listing Models...")
        
        for lid in valid_listings:
            mask = (df['listing_id'] == lid)
            # Train purely on specific data (let fusion handle the safety)
            m = BayesianRidge()
            m.fit(train_matrix[mask], y[mask])
            self.listing_models[str(lid)] = m

        logger.info("Hierarchy Training Complete.")

    def predict(self, context_dict: dict, price: float) -> DemandPrediction:
        """
        Predicts using Bayesian Model Averaging (Fusion).
        Final = (w1*Pred1 + w2*Pred2) / (w1 + w2)
        Where w = 1 / Variance
        """
        lid = str(context_dict.get('listing_id'))
        hood = context_dict.get('neighborhood')
        
        # Input Vector
        x_raw = np.array([[context_dict[f] for f in self.feature_names]])
        x_scaled = self.scaler.transform(x_raw)
        log_price = np.log1p(price)
        input_vec = np.hstack([x_scaled, [[log_price]]])
        
        # 1. Get Component Predictions
        preds = []
        
        # A. City Prediction (Always available)
        m_city, s_city = self.city_model.predict(input_vec, return_std=True)
        preds.append({'mu': m_city[0], 'std': s_city[0], 'level': 'city'})
        
        # B. Neighborhood Prediction
        if hood in self.neighborhood_models:
            m_hood, s_hood = self.neighborhood_models[hood].predict(input_vec, return_std=True)
            preds.append({'mu': m_hood[0], 'std': s_hood[0], 'level': 'neighborhood'})
            
        # C. Listing Prediction
        if lid in self.listing_models:
            m_list, s_list = self.listing_models[lid].predict(input_vec, return_std=True)
            preds.append({'mu': m_list[0], 'std': s_list[0], 'level': 'listing'})
            
        # 2. Fuse (Inverse Variance Weighting)
        # Weight = 1 / (std^2 + epsilon)
        epsilon = 1e-6
        
        numerator = 0.0
        denominator = 0.0
        
        # Determine Source Level for logging (Logic: deepest available)
        source_level = preds[-1]['level'] 
        
        for p in preds:
            # We penalize levels slightly to favor specificity? 
            # No, trusting pure math: precision weighting.
            variance = p['std'] ** 2
            weight = 1.0 / (variance + epsilon)
            
            numerator += p['mu'] * weight
            denominator += weight
            
        final_mu = numerator / denominator
        final_var = 1.0 / denominator
        final_std = np.sqrt(final_var)
        
        # Clip
        final_mu = np.clip(final_mu, 0.0, 1.0)
        
        return DemandPrediction(prob=final_mu, std_dev=final_std, source_level=f"Fused({source_level})")