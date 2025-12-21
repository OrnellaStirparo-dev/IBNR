import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

def plot_model_comparison_metrics(comparison_df: pd.DataFrame):
    """
    Plot comparison metrics (MAE, RMSE, MAPE) for all models.
    
    Args:
        comparison_df: DataFrame with columns ['model', 'mae', 'rmse', 'mape', 'r2']
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison Metrics', fontsize=16, fontweight='bold')
    
    metrics = ['mae', 'rmse', 'mape', 'r2']
    titles = ['Mean Absolute Error (Lower is Better)', 
              'Root Mean Squared Error (Lower is Better)',
              'Mean Absolute Percentage Error (Lower is Better)',
              'R² Score (Higher is Better)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Filter out NaN values
        data = comparison_df[['model', metric]].dropna()
        
        if len(data) > 0:
            sns.barplot(data=data, x='model', y=metric, ax=ax, palette='viridis')
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel(metric.upper(), fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    plt.show()


def plot_ultimate_claims_comparison(summary_df: pd.DataFrame):
    """
    Plot ultimate claims estimates from all models by accident year.
    
    Args:
        summary_df: DataFrame with accident_year and model columns
    """
    # Get model columns (exclude accident_year, latest_reported, and IBNR columns)
    model_cols = [col for col in summary_df.columns 
                  if col not in ['accident_year', 'latest_reported'] 
                  and not col.endswith('_IBNR')]
    
    if len(model_cols) == 0:
        print("No model columns found for comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Ultimate claims by accident year
    ax1 = axes[0]
    x = np.arange(len(summary_df))
    width = 0.8 / (len(model_cols) + 1)
    
    for i, model in enumerate(model_cols):
        offset = (i - len(model_cols) / 2) * width
        ax1.bar(x + offset, summary_df[model], width, label=model, alpha=0.8)
    
    # Add latest reported as reference
    ax1.plot(x, summary_df['latest_reported'], 'ko-', label='Latest Reported', 
            linewidth=2, markersize=8)
    
    ax1.set_xlabel('Accident Year', fontsize=12)
    ax1.set_ylabel('Ultimate Claims', fontsize=12)
    ax1.set_title('Ultimate Claims Estimates by Model', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df['accident_year'].astype(int), rotation=45)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total ultimate and IBNR
    ax2 = axes[1]
    totals = []
    ibnr_totals = []
    labels = []
    
    for model in model_cols:
        if summary_df[model].notna().any():
            totals.append(summary_df[model].sum())
            ibnr_col = f'{model}_IBNR'
            if ibnr_col in summary_df.columns:
                ibnr_totals.append(summary_df[ibnr_col].sum())
            else:
                ibnr_totals.append(summary_df[model].sum() - summary_df['latest_reported'].sum())
            labels.append(model)
    
    x_pos = np.arange(len(labels))
    width = 0.35
    
    ax2.bar(x_pos - width/2, totals, width, label='Total Ultimate', alpha=0.8, color='steelblue')
    ax2.bar(x_pos + width/2, ibnr_totals, width, label='Total IBNR', alpha=0.8, color='coral')
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Amount', fontsize=12)
    ax2.set_title('Total Ultimate Claims and IBNR Reserve', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_ibnr_comparison(summary_df: pd.DataFrame):
    """
    Plot IBNR reserves by accident year for all models.
    
    Args:
        summary_df: DataFrame with accident_year and model_IBNR columns
    """
    # Get IBNR columns
    ibnr_cols = [col for col in summary_df.columns if col.endswith('_IBNR')]
    
    if len(ibnr_cols) == 0:
        print("No IBNR columns found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(summary_df))
    width = 0.8 / len(ibnr_cols)
    
    for i, col in enumerate(ibnr_cols):
        model_name = col.replace('_IBNR', '')
        offset = (i - len(ibnr_cols) / 2) * width
        ax.bar(x + offset, summary_df[col], width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Accident Year', fontsize=12)
    ax.set_ylabel('IBNR Reserve', fontsize=12)
    ax.set_title('IBNR Reserve Estimates by Model and Accident Year', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['accident_year'].astype(int), rotation=45)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


def plot_triangle_with_predictions(triangle_df: pd.DataFrame, 
                                   model_name: str,
                                   predictions: Optional[pd.Series] = None):
    """
    Plot the triangle heatmap with model predictions overlaid.
    
    Args:
        triangle_df: Original triangle DataFrame
        model_name: Name of the model
        predictions: Optional predictions to overlay
    """
    pivot = triangle_df.pivot(index="accident_year", columns="development_year", values="cumulative_claims")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar_kws={'label': 'Cumulative Claims'})
    
    ax.set_title(f'Claims Triangle - {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accident Year', fontsize=12)
    ax.set_xlabel('Development Year', fontsize=12)
    
    plt.tight_layout()
    plt.show()

