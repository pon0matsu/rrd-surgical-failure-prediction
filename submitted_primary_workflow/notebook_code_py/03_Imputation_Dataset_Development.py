# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# %%
!pip install miceforest

# %%
data.columns

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import miceforest as mf

# Load data
data = pd.read_csv('df_ppv_before_split.csv', encoding='UTF-8')

# %%
# Split into training and test data
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['Failure level (0 vs 1 to 3) (6M)'])

# %%
# Exclude ID and target variable
X_train = train_data.drop(columns=['ID', 'Failure level (0 vs 1 to 3) (6M)'])
y_train = train_data['Failure level (0 vs 1 to 3) (6M)']
X_test = test_data.drop(columns=['ID', 'Failure level (0 vs 1 to 3) (6M)'])
y_test = test_data['Failure level (0 vs 1 to 3) (6M)']

# %%
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import miceforest as mf

# Impute missing values in the training data using miceforest
kernel = mf.ImputationKernel(
    X_train,
    datasets=5,
    save_all_iterations=True,
    random_state=0
)
kernel.mice(5)

# Retrieve the imputed training data
imputed_X_train = kernel.complete_data(0)

# Impute missing values in the test data using the model created from the training data
imputed_X_test = kernel.impute_new_data(X_test).complete_data(0)

# Reattach ID and target variable to the imputed data
imputed_train_data = pd.concat([train_data[['ID']], y_train, imputed_X_train], axis=1)
imputed_test_data = pd.concat([test_data[['ID']], y_test, imputed_X_test], axis=1)

# Save to a CSV file
train_data.to_csv('analysis_outputs/train_data_before_imputation.csv', index=False, encoding='utf-8-sig')
imputed_train_data.to_csv('analysis_outputs/train_data_after_imputation.csv', index=False, encoding='utf-8-sig')
test_data.to_csv('analysis_outputs/test_data_before_imputation.csv', index=False, encoding='utf-8-sig')
imputed_test_data.to_csv('analysis_outputs/test_data_after_imputation.csv', index=False, encoding='utf-8-sig')

# %%
# Check whether missing values remain
imputed_train_data.isnull().sum()

# %%
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import miceforest as mf

# Impute missing values in the training data using miceforest
kernel = mf.ImputationKernel(
    X_train,
    datasets=5,
    save_all_iterations=True,
    random_state=0
)
kernel.mice(5)

# Retrieve the imputed training data
imputed_X_train = kernel.complete_data(0)

# Impute missing values in the test data using the model created from the training data
imputed_X_test = kernel.impute_new_data(X_test).complete_data(0)

# Reattach ID and target variable to the imputed data
imputed_train_data = pd.concat([train_data[['ID']], y_train, imputed_X_train], axis=1)
imputed_test_data = pd.concat([test_data[['ID']], y_test, imputed_X_test], axis=1)

# Save to a CSV file
train_data.to_csv('train_data_before_imputation.csv', index=False, encoding='utf-8-sig')
imputed_train_data.to_csv('train_data_after_imputation.csv', index=False, encoding='utf-8-sig')
test_data.to_csv('test_data_before_imputation.csv', index=False, encoding='utf-8-sig')
imputed_test_data.to_csv('test_data_after_imputation.csv', index=False, encoding='utf-8-sig')

# %%
# Set up cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create the directory for saving CSV files
output_dir = 'analysis_outputs'
os.makedirs(output_dir, exist_ok=True)

# Impute missing values in the training and validation data for each fold
X_train_full = train_data.drop(columns=['ID', 'Failure level (0 vs 1 to 3) (6M)'])
y_train_full = train_data['Failure level (0 vs 1 to 3) (6M)']
for fold, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full)):
    fold_train_data = train_data.iloc[train_index]
    fold_val_data = train_data.iloc[val_index]


    # Separate features and target variables for the training and validation data
    X_fold_train = fold_train_data.drop(columns=['ID', 'Failure level (0 vs 1 to 3) (6M)'])
    y_fold_train = fold_train_data['Failure level (0 vs 1 to 3) (6M)']
    X_fold_val = fold_val_data.drop(columns=['ID', 'Failure level (0 vs 1 to 3) (6M)'])
    y_fold_val = fold_val_data['Failure level (0 vs 1 to 3) (6M)']

    # Impute missing values in the training data using miceforest
    kernel = mf.ImputationKernel(
        X_fold_train,
        datasets=5,
        save_all_iterations=True,
        random_state=0
    )
    kernel.mice(5)

    # Retrieve the imputed training data
    imputed_X_fold_train = kernel.complete_data(0)

    # Impute missing values in the validation data using the model created from the training data
    imputed_X_fold_val = kernel.impute_new_data(X_fold_val).complete_data(0)

    # Reattach ID and target variable to the imputed data
    imputed_fold_train_data = pd.concat([fold_train_data[['ID']], y_fold_train, imputed_X_fold_train], axis=1)
    imputed_fold_val_data = pd.concat([fold_val_data[['ID']], y_fold_val, imputed_X_fold_val], axis=1)

    # Save to a CSV file
    fold_train_data.to_csv(f'{output_dir}/fold_{fold}_train_data_before_imputation.csv', index=False, encoding='utf-8-sig')
    imputed_fold_train_data.to_csv(f'{output_dir}/fold_{fold}_train_data_after_imputation.csv', index=False, encoding='utf-8-sig')
    fold_val_data.to_csv(f'{output_dir}/fold_{fold}_val_data_before_imputation.csv', index=False, encoding='utf-8-sig')
    imputed_fold_val_data.to_csv(f'{output_dir}/fold_{fold}_val_data_after_imputation.csv', index=False, encoding='utf-8-sig')

# %%
# Check whether missing values remain
imputed_fold_val_data.isnull().sum()

# %%
# Set up cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create the directory for saving CSV files
output_dir = 'imputation_results'
os.makedirs(output_dir, exist_ok=True)

# Impute missing values in the training and validation data for each fold
X_train_full = train_data.drop(columns=['ID', 'Failure level (0 vs 1 to 3) (6M)'])
y_train_full = train_data['Failure level (0 vs 1 to 3) (6M)']
for fold, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full)):
    fold_train_data = train_data.iloc[train_index]
    fold_val_data = train_data.iloc[val_index]


    # Separate features and target variables for the training and validation data
    X_fold_train = fold_train_data.drop(columns=['ID', 'Failure level (0 vs 1 to 3) (6M)'])
    y_fold_train = fold_train_data['Failure level (0 vs 1 to 3) (6M)']
    X_fold_val = fold_val_data.drop(columns=['ID', 'Failure level (0 vs 1 to 3) (6M)'])
    y_fold_val = fold_val_data['Failure level (0 vs 1 to 3) (6M)']

    # Impute missing values in the training data using miceforest
    kernel = mf.ImputationKernel(
        X_fold_train,
        datasets=5,
        save_all_iterations=True,
        random_state=0
    )
    kernel.mice(5)

    # Retrieve the imputed training data
    imputed_X_fold_train = kernel.complete_data(0)

    # Impute missing values in the validation data using the model created from the training data
    imputed_X_fold_val = kernel.impute_new_data(X_fold_val).complete_data(0)

    # Reattach ID and target variable to the imputed data
    imputed_fold_train_data = pd.concat([fold_train_data[['ID']], y_fold_train, imputed_X_fold_train], axis=1)
    imputed_fold_val_data = pd.concat([fold_val_data[['ID']], y_fold_val, imputed_X_fold_val], axis=1)

    # Save to a CSV file
    fold_train_data.to_csv(f'{output_dir}/fold_{fold}_train_data_before_imputation.csv', index=False, encoding='utf-8-sig')
    imputed_fold_train_data.to_csv(f'{output_dir}/fold_{fold}_train_data_after_imputation.csv', index=False, encoding='utf-8-sig')
    fold_val_data.to_csv(f'{output_dir}/fold_{fold}_val_data_before_imputation.csv', index=False, encoding='utf-8-sig')
    imputed_fold_val_data.to_csv(f'{output_dir}/fold_{fold}_val_data_after_imputation.csv', index=False, encoding='utf-8-sig')

