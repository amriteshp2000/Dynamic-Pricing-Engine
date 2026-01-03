import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PriceEngine_Audit")


@dataclass
class AuditThresholds:
    min_price_std: float = 5.0
    min_bookings: int = 5
    min_variance_coverage: float = 0.20
    max_sparsity_rate: float = 0.90
    min_exposure_days: int = 30   # NEW: minimum exposure to learn anything


class MarketAuditor:
    """
    Module 00: Market Viability & Data Sanity Audit.

    IMPORTANT:
    - Audit respects censoring
    - No price imputation
    - Booking signal is a proxy
    """

    def __init__(self, thresholds: AuditThresholds = AuditThresholds()):
        self.thresh = thresholds

    # --------------------------------------------------
    # 1. Minimal preparation (NO FABRICATION)
    # --------------------------------------------------
    def clean_and_prep(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for audit ONLY.
        Assumes canonical loader output.
        """
        logger.info("Preparing audit dataframe...")

        # Defensive checks
        required_cols = {
            'listing_id',
            'date',
            'price',
            'action_observed',
            'exposed',
            'is_booked_proxy',
            'dow'
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for audit: {missing}")

        return df.copy()

    # --------------------------------------------------
    # 2. Identifiability: price variation WHEN action observed
    # --------------------------------------------------
    def check_identifiability(self, df: pd.DataFrame) -> Dict[str, float]:
        logger.info("Checking Identifiability (Observed Price Variance)...")

        # Only consider days where host revealed a price
        observed = df[df['action_observed']]

        price_std = (
            observed
            .groupby('listing_id')['price']
            .std()
            .dropna()
        )

        valid_count = (price_std > self.thresh.min_price_std).sum()
        total_count = price_std.shape[0]
        pct_valid = valid_count / max(total_count, 1)

        return {
            "total_listings_evaluated": total_count,
            "varying_listings": valid_count,
            "pct_identifiable": pct_valid,
            "status": "PASS" if pct_valid >= self.thresh.min_variance_coverage else "FAIL"
        }

    # --------------------------------------------------
    # 3. Sparsity: effective learning opportunity
    # --------------------------------------------------
    def check_sparsity(self, df: pd.DataFrame) -> Dict[str, float]:
        logger.info("Checking Sparsity (Exposure & Outcome)...")

        # Exposure days per listing
        exposure_counts = df.groupby('listing_id')['exposed'].sum()

        # Proxy bookings only when outcome is observable
        booking_counts = (
            df[df['exposed']]
            .groupby('listing_id')['is_booked_proxy']
            .sum()
        )

        pct_low_exposure = (exposure_counts < self.thresh.min_exposure_days).mean()
        pct_zero_bookings = (booking_counts == 0).mean()

        return {
            "pct_low_exposure": pct_low_exposure,
            "pct_observable_bookings": ( df['action_observed'] & df['is_booked_proxy']).mean(),
            "recommendation": (
                "HIERARCHICAL_BAYES"
                if pct_low_exposure > 0.5
                else "INDEPENDENT_MODELS"
            )
        }

    # --------------------------------------------------
    # 4. Confounding check (proxy, slope-based)
    # --------------------------------------------------
    def check_confounding(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Checking Confounding (Price vs Demand Proxy)...")

        usable = df[df['action_observed'] & df['exposed']]

        if usable.empty:
            return {
                "raw_slope": np.nan,
                "stratified_slope": np.nan,
                "confounding_detected": False,
                "note": "Insufficient observable action-outcome pairs"
            }

        def slope(sub):
            if sub['price'].nunique() < 5:
                return np.nan
            price_bins = pd.qcut(sub['price'], q=5, duplicates='drop')
            rates = sub.groupby(price_bins)['is_booked_proxy'].mean()
            if len(rates) < 3:
                return np.nan
            return np.corrcoef(range(len(rates)), rates)[0, 1]

        raw_slope = slope(usable)

        strat_slopes = []
        for d in range(7):
            s = slope(usable[usable['dow'] == d])
            if not np.isnan(s):
                strat_slopes.append(s)

        strat_slope = np.mean(strat_slopes) if strat_slopes else np.nan

        confounding = (
            raw_slope >= 0 and
            not np.isnan(strat_slope) and
            strat_slope < 0
        )

        return {
            "raw_price_demand_slope": raw_slope,
            "stratified_price_demand_slope": strat_slope,
            "confounding_detected": confounding
        }
