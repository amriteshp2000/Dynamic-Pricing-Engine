import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("PriceEngine_Loader")


def load_and_clean_seattle_data(calendar_path: str, listings_path: str) -> pd.DataFrame:
    """
    Canonical data loader for dynamic pricing experiments.

    DESIGN CONTRACT:
    - No price imputation
    - Explicit exposure, action, outcome flags
    - No row dropped due to censoring
    - Safe to use for demand modeling, bandits, evaluation, benchmarking
    """

    logger.info("Loading raw data...")
    df_cal = pd.read_csv(calendar_path)
    df_list = pd.read_csv(listings_path)

    # --------------------------------------------------
    # 1. Parse date
    # --------------------------------------------------
    df_cal['date'] = pd.to_datetime(df_cal['date'], errors='coerce')

    # --------------------------------------------------
    # 2. Clean price (NO IMPUTATION)
    # --------------------------------------------------
    df_cal['price_raw'] = df_cal['price']

    df_cal['price'] = (
        df_cal['price']
        .astype(str)
        .str.replace(r'[^\d.,-]', '', regex=True)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df_cal['price'] = pd.to_numeric(df_cal['price'], errors='coerce')

    # Action observed only when price is visible
    df_cal['action_observed'] = df_cal['price'].notna()

    # --------------------------------------------------
    # 3. Exposure & outcome semantics (explicit)
    # --------------------------------------------------
    # Exposure: listing was available to be booked
    df_cal['exposed'] = (df_cal['available'].str.lower() == 't')

    # Outcome proxy: booked vs not (AUDIT / RESEARCH PROXY ONLY)
    df_cal['is_booked_proxy'] = (df_cal['available'].str.lower() == 'f').astype(int)

    # Outcome observed only when exposed
    df_cal['outcome_observed'] = df_cal['exposed']

    # --------------------------------------------------
    # 4. Time features (safe, non-leaky)
    # --------------------------------------------------
    df_cal['dow'] = df_cal['date'].dt.dayofweek
    df_cal['week'] = df_cal['date'].dt.isocalendar().week.astype(int)
    df_cal['month'] = df_cal['date'].dt.month

    # --------------------------------------------------
    # 5. Merge listing metadata (immutable / slow-moving only)
    # --------------------------------------------------
    if 'id' in df_list.columns:
        df_list = df_list.rename(columns={'id': 'listing_id'})

    # Minimal, safe columns
    meta_cols = [
        'listing_id',
        'neighbourhood_group_cleansed',
        'neighbourhood',
        'latitude',
        'longitude',
        'room_type',
        'property_type',
        'accommodates',
        'bedrooms',
        'bathrooms'
    ]
    meta_cols = [c for c in meta_cols if c in df_list.columns]

    df = df_cal.merge(df_list[meta_cols], on='listing_id', how='inner')

    # Normalize neighborhood column
    neighborhood_cols = [c for c in df.columns if 'neighbourhood' in c]
    if neighborhood_cols:
        df = df.rename(columns={neighborhood_cols[0]: 'neighborhood'})

    # --------------------------------------------------
    # 6. Sanity logging (no filtering)
    # --------------------------------------------------
    logger.info(f"Total rows loaded: {len(df)}")
    logger.info(f"Action observed rate: {df['action_observed'].mean():.2%}")
    logger.info(f"Exposure rate: {df['exposed'].mean():.2%}")
    logger.info(f"Booked proxy rate: {df['is_booked_proxy'].mean():.2%}")

    return df
