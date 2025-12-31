
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("PriceEngine_Loader")

def load_and_clean_seattle_data(calendar_path: str, listings_path: str) -> pd.DataFrame:
    """
    Robust loader for Seattle Airbnb data with PRICE IMPUTATION.
    Recover prices for booked nights using forward/backward fill.
    """
    logger.info("Loading raw data...")
    df_cal = pd.read_csv(calendar_path)
    df_list = pd.read_csv(listings_path)
    
    # --- 1. Clean Price Column (Strings to Float) ---
    df_cal['price'] = (
        df_cal['price']
        .astype(str)
        .str.replace(r'[$,]', '', regex=True)
        .str.strip()
    )
    df_cal['price'] = pd.to_numeric(df_cal['price'], errors='coerce')
    
    # --- 2. CRITICAL FIX: Impute Prices for Booked Nights ---
    # Sort by listing and date to ensure continuity
    df_cal['date'] = pd.to_datetime(df_cal['date'])
    df_cal = df_cal.sort_values(['listing_id', 'date'])
    
    # Forward Fill then Backward Fill prices WITHIN each listing
    # This assumes if it's booked today, the price was likely the same as yesterday
    df_cal['price'] = df_cal.groupby('listing_id')['price'].ffill().bfill()
    
    # Now we drop only if imputation completely failed (host never set a price)
    initial_len = len(df_cal)
    df_cal = df_cal.dropna(subset=['price'])
    logger.info(f"Recovered prices. Kept {len(df_cal)} / {initial_len} rows.")

    # --- 3. Create Demand Signal ---
    # available='f' means it WAS booked.
    df_cal['is_booked'] = (df_cal['available'].str.lower() == 'f').astype(int)
    
    # --- 4. Merge Listing Metadata ---
    if 'id' in df_list.columns:
        df_list = df_list.rename(columns={'id': 'listing_id'})
        
    cols_to_keep = ['listing_id', 'neighbourhood_group_cleansed']
    if 'neighbourhood_group_cleansed' not in df_list.columns:
         cols_to_keep = ['listing_id', 'neighbourhood']
         
    df_merged = df_cal.merge(df_list[cols_to_keep], on='listing_id', how='inner')
    
    # Rename for consistency
    neighbor_col = [c for c in df_merged.columns if 'neighbourhood' in c][0]
    df_merged = df_merged.rename(columns={neighbor_col: 'neighborhood'})
    
    # Check if we have both 0s and 1s now
    booked_count = df_merged['is_booked'].sum()
    logger.info(f"Total Booked Nights Recovered: {booked_count}")
    
    return df_merged