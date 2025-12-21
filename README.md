# IBNR Estimation Project
Thesis repo for IBNR (Incurred But Not Reported) claims estimation using traditional actuarial methods and machine learning.

## Available Models

This repository includes the following IBNR estimation models:

### Traditional Actuarial Methods
1. **Chain Ladder** (`src/traditional/chainladder_model.py`)
   - Classic chain ladder method using the `chainladder` library
   - Estimates ultimate claims based on development factors

2. **Bornhuetter-Ferguson** (`src/traditional/bf_model.py`)
   - Combines Chain Ladder development factors with prior estimates
   - Useful when you have external information about expected ultimate claims

### Machine Learning Methods
3. **XGBoost** (`src/ml/xgboost_model.py`)
   - Gradient boosting regression model
   - Handles non-linear relationships in claims development

4. **Random Forest** (`src/ml/random_forest.py`)
   - Ensemble tree-based model
   - Provides feature importance insights

5. **Neural Network** (`src/ml/neural_network.py`)
   - Deep learning model using TensorFlow/Keras
   - Multi-layer perceptron for complex pattern recognition

## Data Format

The project expects claims data in Excel format (`ssn_2025_siniestro_casco.xlsx`). The triangle transformer (`src/data/transformer.py`) can handle multiple input formats:

1. **Already in triangle format**: Columns `accident_year`, `development_year`, `cumulative_claims`
2. **Transactional data with dates**: Accident date and development date columns
3. **Transactional data with years**: Accident year and development period columns

The transformer automatically detects common column names:
- Accident year: `año_siniestro`, `accident_year`, `ay`, etc.
- Development year: `development_year`, `dev_year`, `dy`, etc.
- Claims amount: `importe`, `monto`, `claims`, `cumulative_claims`, etc.

## Usage

### Setup

To recreate the virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running All Models

Run the main script to execute all models:

```bash
python main.py
```

This will:
1. Load the Excel file from `data/raw/ssn_2025_siniestro_casco.xlsx`
2. Clean and transform the data into a claims triangle
3. Run all 5 models (Chain Ladder, Bornhuetter-Ferguson, XGBoost, Random Forest, Neural Network)
4. Display results and metrics

### Using Individual Models

You can also import and use individual models:

```python
from src.data.load_data import load_claims_triangle
from src.data.transformer import create_claims_triangle, clean_siniestros_data
from src.traditional.chainladder_model import run_chainladder
from src.traditional.bf_model import run_bornhuetter_ferguson
from src.ml.xgboost_model import run_xgboost
from src.ml.random_forest import run_random_forest
from src.ml.neural_network import run_neural_network

# Load and prepare data
raw_df = load_claims_triangle()
cleaned_df = clean_siniestros_data(raw_df)
triangle_df = create_claims_triangle(cleaned_df)

# Run a specific model
ultimate = run_chainladder(triangle_df.pivot(index='accident_year', columns='development_year', values='cumulative_claims'))
```

## Project Structure

```
IBNR/
├── data/
│   ├── raw/              # Raw Excel files
│   └── processed/        # Processed data
├── src/
│   ├── data/             # Data loading and transformation
│   ├── traditional/      # Traditional actuarial methods
│   ├── ml/               # Machine learning models
│   ├── evaluation/       # Model evaluation and comparison
│   └── visualization/    # Plotting and visualization
├── notebooks/            # Jupyter notebooks for exploration
├── main.py               # Main script to run all models
└── requirements.txt      # Python dependencies
```

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- chainladder: Actuarial triangle methods
- scikit-learn: Machine learning utilities
- xgboost: Gradient boosting
- tensorflow: Neural networks
- matplotlib, seaborn: Visualization
- openpyxl: Excel file reading