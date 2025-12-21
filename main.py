from src.data.load_data import load_claims_triangle
from src.data.transformer import create_claims_triangle, clean_siniestros_data
from src.traditional.chainladder_model import run_chainladder
from src.traditional.bf_model import run_bornhuetter_ferguson
from src.ml.xgboost_model import run_xgboost
from src.ml.neural_network import run_neural_network
from src.ml.random_forest import run_random_forest
from src.evaluation.comparison import create_model_summary, compare_models_results
from src.visualization.comparison import (
    plot_model_comparison_metrics,
    plot_ultimate_claims_comparison,
    plot_ibnr_comparison
)
from src.visualization.triangle_viz import plot_triangle
import pandas as pd

def main():
    """
    Main function to run all IBNR estimation models.
    
    This function:
    1. Loads the raw Excel data
    2. Transforms it into a claims triangle
    3. Runs all available models (Chain Ladder, Bornhuetter-Ferguson, ML models)
    """
    print("=" * 60)
    print("IBNR ESTIMATION - Running All Models")
    print("=" * 60)
    
    # Load your claims data
    # To load a specific sheet, use:
    #   raw_df = load_claims_triangle(sheet_name=1)  # Load second sheet (0-based index)
    #   raw_df = load_claims_triangle(sheet_name="Sheet2")  # Load sheet by name
    print("\n📂 Loading raw data from Excel...")
    raw_df = load_claims_triangle()  # Default: first sheet (index 0)
    
    print(f"   Raw data shape: {raw_df.shape}")
    print(f"   Columns: {raw_df.columns.tolist()}")
    
    # Clean the data
    print("\n🧹 Cleaning data...")
    cleaned_df = clean_siniestros_data(raw_df)
    
    # Transform to triangle format
    print("\n🔺 Creating claims triangle...")
    triangle_df = None
    try:
        triangle_df = create_claims_triangle(cleaned_df)
        print(f"   Triangle shape: {triangle_df.shape}")
        print(f"   Sample data:\n{triangle_df.head(10)}")
    except Exception as e:
        print(f"❌ Error creating triangle: {e}")
        print("\n💡 The data format might be different. Please check:")
        print(f"   - Columns in your data: {cleaned_df.columns.tolist()}")
        print(f"   - First few rows:\n{cleaned_df.head()}")
        print("\n   You may need to specify column names manually:")
        print("   triangle_df = create_claims_triangle(cleaned_df,")
        print("       accident_year_col='your_accident_year_column',")
        print("       development_year_col='your_dev_year_column',")
        print("       claims_col='your_claims_column')")
        raise ValueError("Could not create triangle. Please check your data format and column names.")
    
    print("\n" + "=" * 60)
    print("TRADITIONAL METHODS")
    print("=" * 60)
    
    # 1. Chain Ladder
    print("\n1️⃣  Running Chain Ladder...")
    try:
        # For chainladder library, we need to pivot the triangle
        triangle_pivot = triangle_df.pivot(
            index='accident_year', 
            columns='development_year', 
            values='cumulative_claims'
        )
        ultimate_cl = run_chainladder(triangle_pivot)
        print("✅ Chain Ladder completed")
        print(f"   Ultimate Claims:\n{ultimate_cl}")
    except Exception as e:
        print(f"❌ Chain Ladder failed: {e}")
        ultimate_cl = None
    
    # 2. Bornhuetter-Ferguson
    print("\n2️⃣  Running Bornhuetter-Ferguson...")
    try:
        triangle_pivot = triangle_df.pivot(
            index='accident_year', 
            columns='development_year', 
            values='cumulative_claims'
        )
        ultimate_bf = run_bornhuetter_ferguson(triangle_pivot)
        print("✅ Bornhuetter-Ferguson completed")
        print(f"   Ultimate Claims:\n{ultimate_bf}")
    except Exception as e:
        print(f"❌ Bornhuetter-Ferguson failed: {e}")
        ultimate_bf = None
    
    print("\n" + "=" * 60)
    print("MACHINE LEARNING METHODS")
    print("=" * 60)
    
    # 3. XGBoost
    print("\n3️⃣  Running XGBoost...")
    try:
        model_xgb = run_xgboost(triangle_df, target="cumulative_claims")
        print("✅ XGBoost completed")
    except Exception as e:
        print(f"❌ XGBoost failed: {e}")
        model_xgb = None
    
    # 4. Random Forest
    print("\n4️⃣  Running Random Forest...")
    try:
        model_rf = run_random_forest(triangle_df, target="cumulative_claims")
        print("✅ Random Forest completed")
    except Exception as e:
        print(f"❌ Random Forest failed: {e}")
        model_rf = None
    
    # 5. Neural Network
    print("\n5️⃣  Running Neural Network...")
    try:
        model_nn, history, scaler = run_neural_network(triangle_df, target="cumulative_claims")
        print("✅ Neural Network completed")
    except Exception as e:
        print(f"❌ Neural Network failed: {e}")
        model_nn = None
    
    print("\n" + "=" * 60)
    print("COLLECTING RESULTS FOR COMPARISON")
    print("=" * 60)
    
    # Collect all model results
    model_results = {
        "Chain Ladder": ultimate_cl,
        "Bornhuetter-Ferguson": ultimate_bf,
        # ML models don't return ultimate directly, but we'll handle them in summary
    }
    
    # Create summary DataFrame
    print("\n📊 Creating model comparison summary...")
    try:
        summary_df = create_model_summary(triangle_df, model_results)
        print("✅ Summary created")
        print(f"\nSummary DataFrame:\n{summary_df.to_string()}")
    except Exception as e:
        print(f"⚠️  Could not create full summary: {e}")
        summary_df = None
    
    # Create comparison metrics (if we have actual vs predicted)
    print("\n📈 Calculating comparison metrics...")
    try:
        results_dict = {
            "Chain Ladder": pd.DataFrame(ultimate_cl) if ultimate_cl is not None else None,
            "Bornhuetter-Ferguson": pd.DataFrame(ultimate_bf) if ultimate_bf is not None else None,
        }
        comparison_df = compare_models_results(results_dict, triangle_df)
        if len(comparison_df) > 0:
            print("✅ Comparison metrics calculated")
            print(f"\nComparison Metrics:\n{comparison_df.to_string()}")
        else:
            comparison_df = None
    except Exception as e:
        print(f"⚠️  Could not calculate comparison metrics: {e}")
        comparison_df = None
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Plot 1: Triangle heatmap
    print("\n📊 Plotting claims triangle...")
    try:
        plot_triangle(triangle_df)
    except Exception as e:
        print(f"⚠️  Could not plot triangle: {e}")
    
    # Plot 2: Ultimate claims comparison
    if summary_df is not None and len(summary_df) > 0:
        print("\n📊 Plotting ultimate claims comparison...")
        try:
            plot_ultimate_claims_comparison(summary_df)
        except Exception as e:
            print(f"⚠️  Could not plot ultimate comparison: {e}")
    
    # Plot 3: IBNR comparison
    if summary_df is not None and len(summary_df) > 0:
        print("\n📊 Plotting IBNR comparison...")
        try:
            plot_ibnr_comparison(summary_df)
        except Exception as e:
            print(f"⚠️  Could not plot IBNR comparison: {e}")
    
    # Plot 4: Model metrics comparison
    if comparison_df is not None and len(comparison_df) > 0:
        print("\n📊 Plotting model metrics comparison...")
        try:
            plot_model_comparison_metrics(comparison_df)
        except Exception as e:
            print(f"⚠️  Could not plot metrics comparison: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ All models have been executed")
    if triangle_df is not None and 'accident_year' in triangle_df.columns:
        print(f"   Triangle data: {len(triangle_df)} records")
        print(f"   Accident years: {triangle_df['accident_year'].min():.0f} - {triangle_df['accident_year'].max():.0f}")
        print(f"   Development years: 1 - {triangle_df['development_year'].max():.0f}")
    else:
        print(f"   ⚠️  Triangle data format issue - {len(cleaned_df) if triangle_df is None else len(triangle_df)} records")
        print(f"   Available columns: {cleaned_df.columns.tolist() if triangle_df is None else triangle_df.columns.tolist()}")
    
    if summary_df is not None:
        print(f"\n📊 Model Results Summary:")
        print(f"   Total Ultimate Claims by Model:")
        for col in summary_df.columns:
            if col not in ['accident_year', 'latest_reported'] and not col.endswith('_IBNR'):
                total = summary_df[col].sum()
                if not pd.isna(total):
                    print(f"     {col}: {total:,.2f}")
        
        print(f"\n   Total IBNR Reserve by Model:")
        for col in summary_df.columns:
            if col.endswith('_IBNR'):
                model_name = col.replace('_IBNR', '')
                total = summary_df[col].sum()
                if not pd.isna(total):
                    print(f"     {model_name}: {total:,.2f}")
    
    print("\n✅ All visualizations have been generated!")
    print("   Check the popup windows to view the comparison charts.")

if __name__ == "__main__":
    main()
