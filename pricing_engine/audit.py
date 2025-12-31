
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import logging

# Setup structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PriceEngine_Audit")

@dataclass
class AuditThresholds:
    min_price_std: float = 5.0          # Minimum price standard deviation to claim "variation" exists
    min_bookings: int = 5               # Minimum bookings to not be considered "cold start"
    min_variance_coverage: float = 0.20 # Min % of listings that must have price variance
    max_sparsity_rate: float = 0.90     # Max % of listings allowed to be cold before forcing Hierarchical models

class MarketAuditor:
    """
    Module 00: Data Audit & Market Sanity.
    Strictly checks if causal pricing is viable on the provided dataset.
    """
    
    def __init__(self, thresholds: AuditThresholds = AuditThresholds()):
        self.thresh = thresholds

    def clean_and_prep(self, calendar_df: pd.DataFrame, listings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges and cleans raw Airbnb-style data.
        """
        logger.info("Cleaning and merging data...")
        
        # 1. Clean Price (remove $ and ,)
        if calendar_df['price'].dtype == 'O':
            calendar_df['price'] = calendar_df['price'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)
            
        # 2. Define Demand Proxy (available='f' -> booked=1)
        # Note: This assumes blocked days are bookings. In prod, we need real reservation data.
        # For InsideAirbnb, 'f' is the standard proxy.
        calendar_df['is_booked'] = (calendar_df['available'] == 'f').astype(int)
        
        # 3. Dates
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        calendar_df['dow'] = calendar_df['date'].dt.dayofweek
        
        # 4. Merge Neighborhoods (critical for Hierarchical grouping)
        # Ensure listing_id is matching type
        calendar_df['listing_id'] = calendar_df['listing_id'].astype(int)
        listings_df['id'] = listings_df['id'].astype(int)
        
        # Select only relevant columns to save memory
        listings_sub = listings_df[['id', 'neighbourhood_cleansed']]
        
        df = calendar_df.merge(listings_sub, left_on='listing_id', right_on='id', how='inner')
        
        logger.info(f"Data Prep Complete. Rows: {len(df)}")
        return df

    def check_identifiability(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Checks if enough hosts change their prices to learn elasticity.
        """
        logger.info("Checking Identifiability (Price Variance)...")
        
        # Calculate std dev of price per listing
        price_std = df.groupby('listing_id')['price'].std()
        
        # Count how many have 'valid' variation
        valid_count = (price_std > self.thresh.min_price_std).sum()
        total_count = len(price_std)
        pct_valid = valid_count / total_count
        
        return {
            "total_listings": total_count,
            "varying_listings": valid_count,
            "pct_identifiable": pct_valid,
            "status": "PASS" if pct_valid >= self.thresh.min_variance_coverage else "FAIL"
        }

    def check_sparsity(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Checks cold-start prevalence.
        """
        logger.info("Checking Sparsity (Cold Start)...")
        
        # Count bookings per listing
        booking_counts = df.groupby('listing_id')['is_booked'].sum()
        
        pct_cold = (booking_counts < self.thresh.min_bookings).mean()
        pct_zero = (booking_counts == 0).mean()
        
        return {
            "pct_cold_start": pct_cold,
            "pct_zero_bookings": pct_zero,
            "recommendation": "HIERARCHICAL_BAYES" if pct_cold > 0.5 else "INDEPENDENT_MODELS"
        }

    def check_confounding(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Checks for Simpson's Paradox (High Price correlating with High Demand due to weekends).
        """
        logger.info("Checking Confounding (Simpson's Paradox)...")
        
        # 1. Raw Correlation
        raw_corr = df['price'].corr(df['is_booked'])
        
        # 2. Stratified Correlation (Simple Stratification by Day of Week)
        # We average the correlation found within each day of the week
        stratified_corrs = []
        for d in range(7):
            sub = df[df['dow'] == d]
            if len(sub) > 100 and sub['price'].std() > 0:
                c = sub['price'].corr(sub['is_booked'])
                if not np.isnan(c):
                    stratified_corrs.append(c)
        
        avg_strat_corr = np.mean(stratified_corrs) if stratified_corrs else 0.0
        
        # If Raw is positive (or close to 0) but Stratified is negative, we have confounding
        simpsons = (raw_corr > -0.05) and (avg_strat_corr < -0.05)
        
        return {
            "raw_correlation_price_demand": raw_corr,
            "stratified_correlation_price_demand": avg_strat_corr,
            "confounding_detected": simpsons
        }