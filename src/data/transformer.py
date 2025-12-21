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
    
    # Remove rows that are clearly headers/descriptions (all non-numeric or mostly text)
    # Check first few rows - if they're all text, they're likely headers
    rows_to_drop = []
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        # If row has mostly non-numeric or very long text values, it's likely a header
        text_count = sum(1 for val in row if isinstance(val, str) and len(str(val)) > 50)
        if text_count > len(row) * 0.5:
            rows_to_drop.append(df.index[i])
            print(f"   Removed header/description row {i+1}")
    if rows_to_drop:
        df = df.drop(rows_to_drop)
        df = df.reset_index(drop=True)
    
    # Check if data is in pivot/table format (rows = accident years, columns = development periods)
    # This is common in Excel files where the triangle is already structured
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Also check if index contains years
    index_has_years = False
    if df.index.dtype == 'object' or df.index.dtype == 'int64':
        try:
            index_years = pd.to_numeric(df.index, errors='coerce')
            if index_years.notna().sum() > len(df) * 0.5:
                index_has_years = True
        except:
            pass
    
    # If we have mostly numeric columns and one non-numeric (likely accident year column)
    # OR if the index contains years
    if (len(numeric_cols) >= 1 and len(non_numeric_cols) <= 2) or index_has_years:
        print("🔍 Detected pivot table format - attempting to convert...")
        try:
            # Try to identify accident year column
            if index_has_years:
                # Use index as accident years
                df['accident_year'] = pd.to_numeric(df.index, errors='coerce')
                print("   Using index as accident years")
            elif len(non_numeric_cols) > 0:
                # First non-numeric column is likely accident year
                acc_col = non_numeric_cols[0]
                # Try to extract year from first column
                if df[acc_col].dtype == 'object':
                    # Try to convert to numeric (might be years as strings)
                    df['accident_year'] = pd.to_numeric(df[acc_col], errors='coerce')
                    if df['accident_year'].isna().all() or df['accident_year'].isna().sum() > len(df) * 0.5:
                        # If that fails, try to extract year from strings
                        year_extracted = df[acc_col].astype(str).str.extract(r'(\d{4})')[0]
                        df['accident_year'] = pd.to_numeric(year_extracted, errors='coerce')
                else:
                    df['accident_year'] = pd.to_numeric(df[acc_col], errors='coerce')
                print(f"   Using column '{acc_col}' as accident years")
            else:
                # No non-numeric columns - try to use row numbers or first numeric column
                # Assume rows are sequential years starting from some base year
                # This is a fallback - user should specify
                print("   ⚠️  Could not identify accident year column automatically")
                raise ValueError("Cannot auto-detect accident years in pivot format")
            
            if df['accident_year'].isna().all():
                raise ValueError("Could not extract accident years")
            
            # Development years are the numeric column names
            triangle_list = []
            for idx, row in df.iterrows():
                ay = df.loc[idx, 'accident_year']
                if pd.isna(ay):
                    continue
                for col_idx, col in enumerate(numeric_cols):
                    # Try to extract development year from column name
                    dev_year_str = str(col)
                    # Try to find number in column name
                    dev_match = pd.Series([dev_year_str]).str.extract(r'(\d+)')[0]
                    if len(dev_match) > 0 and not dev_match.isna().iloc[0]:
                        try:
                            dev_year = int(dev_match.iloc[0])
                        except:
                            dev_year = col_idx + 1
                    else:
                        # Use column index as development year
                        dev_year = col_idx + 1
                    
                    value = row[col]
                    if pd.notna(value) and pd.notna(ay):
                        triangle_list.append({
                            'accident_year': int(ay),
                            'development_year': dev_year,
                            'cumulative_claims': float(value)
                        })
            
            if len(triangle_list) > 0:
                result_df = pd.DataFrame(triangle_list)
                # Calculate cumulative if needed (might already be cumulative)
                result_df = result_df.sort_values(['accident_year', 'development_year'])
                print(f"✅ Converted pivot table to triangle format: {len(result_df)} records")
                return result_df[['accident_year', 'development_year', 'cumulative_claims']]
        except Exception as e:
            print(f"⚠️  Pivot table conversion failed: {e}")
            # Continue with normal processing
    
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
