"""
Model Evaluation for RRD Surgical Failure Prediction

This script provides evaluation functions for the prediction models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, calibration_curve
from sklearn.calibration import calibration_curve
import shap


def evaluate_risk_stratification(y_true, y_pred_proba, risk_thresholds=[0.1, 0.4, 0.7, 0.9]):
    """
    Evaluate model performance across risk stratification groups
    
    Parameters:
    -----------
    y_true : True labels
    y_pred_proba : Prediction probabilities
    risk_thresholds : List of risk score thresholds
    
    Returns:
    --------
    stratification_results : DataFrame with stratification results
    """
    
    results = []
    
    # Add boundary values
    thresholds = [0.0] + risk_thresholds + [1.0]
    
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i + 1]
        
        # Find patients in this risk group
        mask = (y_pred_proba >= lower) & (y_pred_proba < upper)
        
        if mask.sum() > 0:
            group_true = y_true[mask]
            failure_rate = group_true.mean()
            
            results.append({
                'Risk Score Range': f'[{lower:.1f}, {upper:.1f})',
                'N Patients': mask.sum(),
                'N Failures': group_true.sum(),
                'Failure Rate': failure_rate,
                'Percentage of Total': mask.sum() / len(y_true) * 100
            })
    
    return pd.DataFrame(results)


def plot_roc_curves(models_results, save_path=None):
    """
    Plot ROC curves for multiple models
    
    Parameters:
    -----------
    models_results : Dictionary with model results
    save_path : Path to save figure
    """
    
    plt.figure(figsize=(8, 6))
    
    for name, results in models_results.items():
        y_true = results['y_true']
        y_pred_proba = results['predictions']
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_calibration_curves(models_results, n_bins=10, save_path=None):
    """
    Plot calibration curves for multiple models
    
    Parameters:
    -----------
    models_results : Dictionary with model results
    n_bins : Number of calibration bins
    save_path : Path to save figure
    """
    
    plt.figure(figsize=(8, 6))
    
    for name, results in models_results.items():
        y_true = results['y_true']
        y_pred_proba = results['predictions']
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        plt.plot(mean_predicted_value, fraction_of_positives, 
                marker='o', label=name)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_shap_analysis(model, X_train, X_test, feature_names=None):
    """
    Generate SHAP analysis for model interpretability
    
    Parameters:
    -----------
    model : Trained model
    X_train : Training data for SHAP explainer
    X_test : Test data to explain
    feature_names : List of feature names
    
    Returns:
    --------
    shap_values : SHAP values
    explainer : SHAP explainer object
    """
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model, X_train)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # If binary classification, take values for positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return shap_values, explainer


def create_performance_summary(models_results):
    """
    Create summary table of model performance metrics
    
    Parameters:
    -----------
    models_results : Dictionary with model results
    
    Returns:
    --------
    summary_df : DataFrame with performance summary
    """
    
    summary_data = []
    
    for name, results in models_results.items():
        metrics = results['metrics']
        
        summary_data.append({
            'Model': name,
            'AUROC': f"{metrics['auroc']:.3f}",
            'Sensitivity': f"{metrics['sensitivity']:.3f}",
            'Specificity': f"{metrics['specificity']:.3f}",
            'PPV': f"{metrics['ppv']:.3f}",
            'NPV': f"{metrics['npv']:.3f}"
        })
    
    return pd.DataFrame(summary_data)