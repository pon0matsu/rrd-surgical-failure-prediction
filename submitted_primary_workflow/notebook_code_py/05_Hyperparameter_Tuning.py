# %%
!pip install imblearn

# %%
# loading and preparing the data
# Load the imputed training dataset
df_train = pd.read_csv('analysis_outputs/train_data_after_imputation.csv')
df_test = pd.read_csv('analysis_outputs/test_data_after_imputation.csv')

df_rfecv = pd.read_csv('analysis_outputs/rfecv_features.csv', header=None, encoding='utf-8')
rfecv_features = df_rfecv[0].tolist()

print("CSVから読み込んだ特徴量リスト:")
print(rfecv_features)

X_tr = df_train[rfecv_features]
X_test_final = df_test[rfecv_features]
y_tr = df_train['Failure level (0 vs 1 to 3) (6M)']
y_test_final = df_test['Failure level (0 vs 1 to 3) (6M)']
columns = X_tr.columns

# %%
# loading and preparing the data
# Load the imputed training dataset
df_train = pd.read_csv('train_data_after_imputation.csv')
df_test = pd.read_csv('test_data_after_imputation.csv')

df_rfecv = pd.read_csv('rfecv_features.csv', header=None, encoding='utf-8')
rfecv_features = df_rfecv[0].tolist()

print("CSVから読み込んだ特徴量リスト:")
print(rfecv_features)

X_tr = df_train[rfecv_features]
X_test_final = df_test[rfecv_features]
y_tr = df_train['Failure level (0 vs 1 to 3) (6M)']
y_test_final = df_test['Failure level (0 vs 1 to 3) (6M)']
columns = X_tr.columns

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler
import csv

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42)
}

# Define the hyperparameter grid for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['saga'],
        'penalty': ['none']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

# Dictionary to store hyperparameter tuning results
grid_search_results = {}

for model_name, params in param_grids.items():
    model = models[model_name]
    grid_search = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.3, random_state=42, stratify=y_tr)

    # Create an instance for undersampling
    rus = RandomUnderSampler(random_state=42)

    # Undersample the data
    X_rus, y_rus = rus.fit_resample(X_train, y_train)

    grid_search.fit(X_rus, y_rus)
    test_score = grid_search.score(X_test, y_test)

    # Store results
    grid_search_results[model_name] = {
        'Best Parameters': grid_search.best_params_,
        'Best Score': grid_search.best_score_,
        'Test Score': test_score,
        'Best Estimator': grid_search.best_estimator_
    }

    # Update the model with the best parameters
    models[model_name] = grid_search.best_estimator_

# Output results to a CSV file
output_csv = 'analysis_outputs/CV_5FOLD_UNDERSAMPLING/grid_search_results.csv'

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Best Parameters', 'Best Score', 'Test Score'])

    for model_name, results in grid_search_results.items():
        writer.writerow([
            model_name,
            results['Best Parameters'],
            results['Best Score'],
            results['Test Score']
        ])

print('CSVファイルにグリッドサーチの結果を出力しました。')

# %%


