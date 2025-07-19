"""
Feature Selection for RRD Surgical Failure Prediction

This script demonstrates the feature selection process using RFECV
(Recursive Feature Elimination with Cross-Validation) as described in the paper.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def perform_feature_selection(X, y, cv_folds=5, random_state=42):
    """
    Perform feature selection using RFECV with Random Forest
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    rfecv : RFECV object
        Fitted RFECV object
    """
    
    # Initialize Random Forest classifier
    clf_rf = RandomForestClassifier(random_state=random_state)
    
    # Configure RFECV with stratified cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    rfecv = RFECV(
        estimator=clf_rf,
        step=1,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    # Fit RFECV
    rfecv.fit(X, y)
    
    # Get selected features
    selected_features = X.columns[rfecv.support_].tolist()
    
    print(f'Optimal number of features: {rfecv.n_features_}')
    print(f'Selected features: {len(selected_features)}')
    
    return selected_features, rfecv


def get_feature_importance(X, y, random_state=42):
    """
    Calculate feature importance using Random Forest
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    feature_importance_df : pandas.DataFrame
        DataFrame with features and their importance scores
    """
    
    # Train Random Forest
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X, y)
    
    # Get feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return feature_importance_df


# Example usage (commented out for library use)
"""
# Load data
df_train = pd.read_csv('train_data_after_imputation.csv')
X = df_train.drop(['ID', 'Failure level (0 vs 1 to 3) (6M)'], axis=1)
y = df_train['Failure level (0 vs 1 to 3) (6M)']

# Perform feature selection
selected_features, rfecv = perform_feature_selection(X, y)

# Get feature importance
feature_importance_df = get_feature_importance(X, y)
"""