"""
Data Preprocessing for RRD Surgical Failure Prediction

This script demonstrates the preprocessing steps used in the paper,
including handling missing values and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def preprocess_clinical_data(df):
    """
    Preprocess clinical data for RRD prediction
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw clinical data
        
    Returns:
    --------
    df_processed : pandas.DataFrame
        Preprocessed data
    """
    
    df_processed = df.copy()
    
    # Convert visual acuity to LogMAR
    va_mapping = {
        'CF': 2.1,  # Counting fingers
        'HM': 2.4,  # Hand motion
        'LP': 2.7,  # Light perception
        'NLP': 3.0  # No light perception
    }
    
    # Feature engineering
    # Create hypotony feature (IOP < 5 mmHg)
    if '術前所見_眼圧_眼圧' in df_processed.columns:
        df_processed['Hypotony_5mmHg'] = (df_processed['術前所見_眼圧_眼圧'] < 5).astype(int)
    
    # Create high myopia feature (axial length ≥ 26mm)
    if 'V22a眼軸長' in df_processed.columns:
        df_processed['V22a眼軸長(26mm以上)'] = (df_processed['V22a眼軸長'] >= 26).astype(int)
    
    return df_processed


def handle_missing_values(X_train, X_test=None, strategy='mean'):
    """
    Handle missing values using imputation
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame, optional
        Test features
    strategy : str
        Imputation strategy ('mean', 'median', 'most_frequent')
        
    Returns:
    --------
    X_train_imputed : pandas.DataFrame
        Training data with imputed values
    X_test_imputed : pandas.DataFrame or None
        Test data with imputed values (if provided)
    """
    
    # Separate numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns
    
    # Impute numeric columns
    if len(numeric_cols) > 0:
        imputer_num = SimpleImputer(strategy=strategy)
        X_train[numeric_cols] = imputer_num.fit_transform(X_train[numeric_cols])
        
        if X_test is not None:
            X_test[numeric_cols] = imputer_num.transform(X_test[numeric_cols])
    
    # Impute categorical columns
    if len(categorical_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = imputer_cat.fit_transform(X_train[categorical_cols])
        
        if X_test is not None:
            X_test[categorical_cols] = imputer_cat.transform(X_test[categorical_cols])
    
    return X_train if X_test is None else (X_train, X_test)


def encode_categorical_features(df, categorical_columns):
    """
    One-hot encode categorical features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with categorical features
    categorical_columns : list
        List of categorical column names
        
    Returns:
    --------
    df_encoded : pandas.DataFrame
        Data with one-hot encoded features
    """
    
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
    return df_encoded


# Selected features based on RFECV (from the paper)
SELECTED_FEATURES = [
    '患者情報__性別', '術前所見__黄斑剥離', '術前所見_黄斑剥離期間_日', 
    '術前所見_矯正視力_矯正視力', '術前所見_眼圧_眼圧', '術前所見__脈絡膜剥離', 
    '術前所見_裂孔数_個', 'Hypotony_5mmHg', 'V22a眼軸長(26mm以上)',
    '術前所見__主病名1_PVDに伴う弁状裂孔による裂孔原性網膜剥離',
    '術前所見__主病名1_その他の裂孔原性網膜剥離', 
    '術前所見__主病名1_内眼手術後(白内障硝子体)の裂孔原性網膜剥離',
    '術前所見__主病名1_強度近視に伴う黄斑円孔網膜剥離', 
    '術前所見__主病名1_萎縮円孔による裂孔原性網膜剥離',
    '術前所見_最大裂孔大きさ_度_0-30', '術前所見_最大裂孔大きさ_度_30-60', 
    '術前所見_PVR_N/B/C_B', '術前所見_PVR_N/B/C_C', '術前所見_PVR_N/B/C_N',
    '術前所見__最大裂孔位置_上耳側', '術前所見__最大裂孔位置_上鼻側',
    '術前所見__最大裂孔位置_下耳側', '術前所見__最大裂孔位置_下鼻側',
    '術前所見__最大裂孔位置_後極', '術前所見_裂孔形態_種別_円孔',
    '術前所見_裂孔形態_種別_裂孔', '術前所見_裂孔形態_種別_黄斑円孔',
    '術前所見_網膜剥離範囲_現象_1', '術前所見_網膜剥離範囲_現象_2',
    '術前所見_網膜剥離範囲_現象_3', '術前所見_網膜剥離範囲_現象_4',
    '眼手術歴_網膜剥離を除く網膜硝子体手術', '眼手術歴_白内障手術',
    '術前所見__水晶体_有水晶体眼', '術前所見__水晶体_IOL', '初回手術時年齢'
]