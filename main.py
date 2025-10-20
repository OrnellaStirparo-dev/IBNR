from src.data.load_data import load_claims_triangle
from src.traditional.chainladder_model import run_chainladder
from src.ml.xgboost_model import run_xgboost

def main():
    # Load your claims data
    df = load_claims_triangle()

    # Traditional IBNR
    print("Running Chain Ladder...")
    try:
        ultimate = run_chainladder(df)
        print("Ultimate Claims:\n", ultimate)
    except Exception as e:
        print("Chain Ladder failed:", e)

    # Machine Learning model
    print("Running XGBoost model...")
    run_xgboost(df)

if __name__ == "__main__":
    main()
