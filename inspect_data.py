"""
Helper script to inspect the Excel file structure and help identify columns.
Run this to understand your data format before running main.py
"""
from src.data.load_data import load_claims_triangle
from src.data.transformer import clean_siniestros_data
import pandas as pd

def inspect_data():
    print("=" * 60)
    print("DATA INSPECTION TOOL")
    print("=" * 60)
    
    # Load raw data
    print("\n📂 Loading raw data...")
    raw_df = load_claims_triangle()
    
    print(f"\n📊 Raw Data Info:")
    print(f"   Shape: {raw_df.shape}")
    print(f"   Columns: {raw_df.columns.tolist()}")
    print(f"\n   First 20 rows:")
    print(raw_df.head(20).to_string())
    
    print(f"\n   Data types:")
    print(raw_df.dtypes)
    
    print(f"\n   Sample values from each column:")
    for col in raw_df.columns:
        print(f"     {col}: {raw_df[col].head(5).tolist()}")
    
    # Clean data
    print("\n🧹 Cleaning data...")
    cleaned_df = clean_siniestros_data(raw_df)
    
    print(f"\n📊 Cleaned Data Info:")
    print(f"   Shape: {cleaned_df.shape}")
    print(f"   Columns: {cleaned_df.columns.tolist()}")
    print(f"\n   First 20 rows:")
    print(cleaned_df.head(20).to_string())
    
    print(f"\n   Data types:")
    print(cleaned_df.dtypes)
    
    print(f"\n   Numeric columns: {cleaned_df.select_dtypes(include=['number']).columns.tolist()}")
    print(f"   Non-numeric columns: {cleaned_df.select_dtypes(exclude=['number']).columns.tolist()}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_cols = cleaned_df.select_dtypes(exclude=['number']).columns.tolist()
    
    if len(numeric_cols) >= 2 and len(non_numeric_cols) <= 2:
        print("\n✅ Your data appears to be in PIVOT TABLE format")
        print("   (rows = accident years, columns = development periods)")
        print("   The transformer should automatically detect this.")
    elif len(non_numeric_cols) == 0 and len(numeric_cols) >= 3:
        print("\n✅ Your data appears to be in TRANSACTIONAL format")
        print("   (each row is a claim with accident year, dev year, and amount)")
        print("   You may need to specify column names manually.")
    else:
        print("\n⚠️  Data format unclear. You may need to:")
        print("   1. Check if your Excel file has the correct structure")
        print("   2. Specify column names manually in create_claims_triangle()")
        print("   3. Ensure your data has:")
        print("      - Accident year column")
        print("      - Development year/period column")
        print("      - Claims amount column")

if __name__ == "__main__":
    inspect_data()

