# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

output_dir = 'analysis_outputs'  # Specify the output directory

# Process data for each fold
for fold in range(5):
    # Load training and validation data
    train_data = pd.read_csv(f'{output_dir}/fold_{fold}_train_data_after_imputation.csv')

    # Split into features and target variable
    X_train = train_data.drop(columns=['Failure level (0 vs 1 to 3) (6M)'])
    y_train = train_data['Failure level (0 vs 1 to 3) (6M)']

    # Create an instance for undersampling
    rus = RandomUnderSampler(random_state=42)

    # Undersample the data
    X_rus, y_rus = rus.fit_resample(X_train, y_train)

    # Identify data excluded by undersampling
    excluded_idx = [i for i in range(len(X_train)) if i not in rus.sample_indices_]
    X_excluded = X_train.iloc[excluded_idx]
    y_excluded = y_train.iloc[excluded_idx]

    # Combine undersampled data and excluded data
    X_rus['Failure level (0 vs 1 to 3) (6M)'] = y_rus
    X_excluded['Failure level (0 vs 1 to 3) (6M)'] = y_excluded

    # Save undersampled data and excluded data
    X_rus.to_csv(f'{output_dir}/fold_{fold}_undersampled_train_data.csv', index=False)
    X_excluded.to_csv(f'{output_dir}/fold_{fold}_excluded_train_data.csv', index=False)

print("CSVファイルの作成が完了しました。")

# %%
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

output_dir = 'analysis_outputs'  # Specify the output directory

# Load training and validation data
train_data = pd.read_csv('analysis_outputs/train_data_after_imputation.csv')

# Split into features and target variable
X_train = train_data.drop(columns=['Failure level (0 vs 1 to 3) (6M)'])
y_train = train_data['Failure level (0 vs 1 to 3) (6M)']

# Create an instance for undersampling
rus = RandomUnderSampler(random_state=42)

# Undersample the data
X_rus, y_rus = rus.fit_resample(X_train, y_train)

# Identify data excluded by undersampling
excluded_idx = [i for i in range(len(X_train)) if i not in rus.sample_indices_]
X_excluded = X_train.iloc[excluded_idx]
y_excluded = y_train.iloc[excluded_idx]

# Combine undersampled data and excluded data
X_rus['Failure level (0 vs 1 to 3) (6M)'] = y_rus
X_excluded['Failure level (0 vs 1 to 3) (6M)'] = y_excluded

# Save undersampled data and excluded data
X_rus.to_csv(f'{output_dir}/train_data_after_imputation_undersampled.csv', index=False)
X_excluded.to_csv(f'{output_dir}/train_data_after_imputation_excluded.csv', index=False)

print("CSVファイルの作成が完了しました。")

# %%

