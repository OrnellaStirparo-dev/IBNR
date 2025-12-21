import pandas as pd
import numpy as np
from typing import Optional

def clean_siniestros_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize SSN siniestros data."""
    # Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )

    # Drop empty columns
    df = df.dropna(axis=1, how="all")

    # Example: convert numeric columns
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].replace({",": ".", "-": None}, regex=True)

    # Optional: rename specific columns
    df = df.rename(columns={
        "compañía": "compania",
        "ramo": "ramo",
        "importe_siniestros": "importe"
    })

    # Drop empty rows
    df = df.dropna(how="all")

    print(f"🧹 Cleaned data shape: {df.shape}")
    return df


def create_claims_triangle(df: pd.DataFrame, 
                           accident_year_col: Optional[str] = None,
                           development_year_col: Optional[str] = None,
                           claims_col: Optional[str] = None,
                           date_col: Optional[str] = None,
                           accident_date_col: Optional[str] = None) -> pd.DataFrame:
    """
    Transform raw claims data into a cumulative claims triangle format.
    
    This function handles multiple input formats:
    1. Already in triangle format (accident_year, development_year, cumulative_claims)
    2. Transactional data with accident date and development date
    3. Transactional data with accident year and development period
    
    Args:
        df: Raw claims DataFrame
        accident_year_col: Name of accident year column (e.g., 'año_siniestro', 'accident_year')
        development_year_col: Name of development year/period column
        claims_col: Name of claims amount column (e.g., 'importe', 'monto', 'claims')
        date_col: Name of date column if using dates instead of years
        accident_date_col: Name of accident date column if different from date_col
    
    Returns:
        DataFrame with columns: accident_year, development_year, cumulative_claims
    """
    df = df.copy()
    
    # Normalize column names first
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    
    # Check if already in triangle format
    if all(col in df.columns for col in ['accident_year', 'development_year', 'cumulative_claims']):
        print("✅ Data already in triangle format")
        return df[['accident_year', 'development_year', 'cumulative_claims']].copy()
    
    # Auto-detect columns if not provided
    if accident_year_col is None:
        possible_accident_cols = ['año_siniestro', 'accident_year', 'ay', 'año', 'year', 
                                  'periodo_siniestro', 'fecha_siniestro']
        for col in possible_accident_cols:
            if col in df.columns:
                accident_year_col = col
                break
    
    if development_year_col is None:
        possible_dev_cols = ['development_year', 'dev_year', 'dy', 'periodo_desarrollo',
                            'año_desarrollo', 'lag', 'development_period']
        for col in possible_dev_cols:
            if col in df.columns:
                development_year_col = col
                break
    
    if claims_col is None:
        possible_claims_cols = ['importe', 'monto', 'claims', 'cumulative_claims', 
                               'siniestros', 'valor', 'amount']
        for col in possible_claims_cols:
            if col in df.columns:
                claims_col = col
                break
    
    # Handle date-based data
    if date_col or accident_date_col:
        if accident_date_col and accident_date_col in df.columns:
            df['accident_year'] = pd.to_datetime(df[accident_date_col], errors='coerce').dt.year
        elif date_col and date_col in df.columns:
            df['accident_year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
        
        # Calculate development year from dates
        if date_col in df.columns and accident_date_col in df.columns:
            df['accident_date'] = pd.to_datetime(df[accident_date_col], errors='coerce')
            df['development_date'] = pd.to_datetime(df[date_col], errors='coerce')
            df['development_year'] = (df['development_date'].dt.year - df['accident_date'].dt.year) + 1
        elif date_col in df.columns:
            # Assume date_col is development date, need accident date separately
            if accident_date_col:
                df['accident_date'] = pd.to_datetime(df[accident_date_col], errors='coerce')
                df['development_date'] = pd.to_datetime(df[date_col], errors='coerce')
                df['development_year'] = (df['development_date'].dt.year - df['accident_date'].dt.year) + 1
    
    # Use detected or provided columns
    if accident_year_col and accident_year_col in df.columns:
        df['accident_year'] = pd.to_numeric(df[accident_year_col], errors='coerce')
    elif 'accident_year' not in df.columns:
        raise ValueError("Could not identify accident year column. Please specify accident_year_col.")
    
    if development_year_col and development_year_col in df.columns:
        df['development_year'] = pd.to_numeric(df[development_year_col], errors='coerce')
    elif 'development_year' not in df.columns:
        # If no development year, assume all are development year 1 (or calculate from dates)
        print("⚠️  No development year column found. Assuming all claims are at development year 1.")
        df['development_year'] = 1
    
    if claims_col and claims_col in df.columns:
        df['claims'] = pd.to_numeric(df[claims_col], errors='coerce')
    elif 'cumulative_claims' in df.columns:
        df['claims'] = df['cumulative_claims']
    else:
        # Try to find numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df['claims'] = df[numeric_cols[0]]
            print(f"⚠️  Using '{numeric_cols[0]}' as claims column")
        else:
            raise ValueError("Could not identify claims column. Please specify claims_col.")
    
    # Remove rows with missing essential data
    df = df.dropna(subset=['accident_year', 'development_year', 'claims'])
    
    # Aggregate by accident year and development year
    triangle = df.groupby(['accident_year', 'development_year'])['claims'].sum().reset_index()
    
    # Calculate cumulative claims
    triangle = triangle.sort_values(['accident_year', 'development_year'])
    triangle['cumulative_claims'] = triangle.groupby('accident_year')['claims'].cumsum()
    
    # Create full triangle structure (fill missing development years with NaN for upper triangle)
    accident_years = sorted(triangle['accident_year'].unique())
    max_dev_year = triangle['development_year'].max()
    
    # Create complete grid
    full_triangle = []
    for ay in accident_years:
        ay_data = triangle[triangle['accident_year'] == ay]
        for dy in range(1, max_dev_year + 1):
            existing = ay_data[ay_data['development_year'] == dy]
            if len(existing) > 0:
                full_triangle.append({
                    'accident_year': ay,
                    'development_year': dy,
                    'cumulative_claims': existing['cumulative_claims'].iloc[0]
                })
            else:
                # Upper triangle - no data yet
                full_triangle.append({
                    'accident_year': ay,
                    'development_year': dy,
                    'cumulative_claims': np.nan
                })
    
    result = pd.DataFrame(full_triangle)
    
    print(f"✅ Created triangle with {len(accident_years)} accident years and {max_dev_year} development years")
    print(f"   Accident years: {min(accident_years)} - {max(accident_years)}")
    
    return result[['accident_year', 'development_year', 'cumulative_claims']]
