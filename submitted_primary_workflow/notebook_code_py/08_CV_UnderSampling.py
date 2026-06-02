# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
!pip install tabpfn
!pip install shap
!pip install imblearn
!pip install japanize-matplotlib

# %%
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix, roc_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import japanize_matplotlib
import shap
from tabpfn import TabPFNClassifier
from imblearn.under_sampling import RandomUnderSampler
import joblib
import csv

# %%
# Load the list from a CSV file
df_rfecv = pd.read_csv('analysis_outputs/rfecv_features.csv', header=None, encoding='utf-8')
rfecv_features = df_rfecv[0].tolist()

print("CSVから読み込んだ特徴量リスト:")
print(rfecv_features)

# %%
# Set the directory where fold data are saved
output_dir = 'analysis_outputs'

# List for saving evaluation results
accuracy_scores = []
f1_scores = []
roc_auc_scores = []
precision_scores = []
recall_scores = []
sensitivity_scores = []
specificity_scores = []

# Plot settings for drawing ROC curves
plt.figure(figsize=(10, 8))

# Load data for each fold, then train and evaluate the model
for fold in range(5):
    # Load training and validation data
    train_data = pd.read_csv(f'{output_dir}/fold_{fold}_train_data_after_imputation.csv')
    val_data = pd.read_csv(f'{output_dir}/fold_{fold}_val_data_after_imputation.csv')

    # Split into features and target variable using only features selected by RFECV
    X_val = val_data[rfecv_features]
    y_val = val_data['Failure level (0 vs 1 to 3) (6M)']

    # Undersample the data
    train_data_rus = pd.read_csv(f'{output_dir}/fold_{fold}_undersampled_train_data.csv')
    X_rus = train_data_rus[rfecv_features]
    y_rus = train_data_rus['Failure level (0 vs 1 to 3) (6M)']

    # Initialize the custom model
    tabpfn = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)

    # Train the model
    tabpfn.fit(X_rus, y_rus, overwrite_warning=True)

    # Save the model
    joblib_file = f"{output_dir}/CV_5FOLD_UNDERSAMPLING/tabpfn/tabpfn_model_fold_{fold}.pkl"
    joblib.dump(tabpfn, joblib_file)
    print(f"Model for fold {fold} saved to {joblib_file}")

    # Output predicted probabilities for validation data to a CSV file
    y_pred = tabpfn.predict(X_val)
    y_pred_proba = tabpfn.predict_proba(X_val)[:, 1]
    val_data['pred_proba'] = y_pred_proba
    val_data.to_csv(f'{output_dir}/CV_5FOLD_UNDERSAMPLING/tabpfn/Results_UnderSampling_fold_{fold}_val_data_with_pred_proba.csv', index=False)

    # Data excluded by undersampling
    train_data_excluded = pd.read_csv(f'{output_dir}/fold_{fold}_excluded_train_data.csv')
    X_train = train_data_excluded[rfecv_features]
    y_train = train_data_excluded['Failure level (0 vs 1 to 3) (6M)']

    # Calculate predicted probabilities for excluded data, merge them into train_data_excluded, and output to a CSV file
    y_pred_proba_excluded = tabpfn.predict_proba(X_train)[:, 1]
    train_data_excluded['pred_proba'] = y_pred_proba_excluded
    train_data_excluded.to_csv(f'{output_dir}/CV_5FOLD_UNDERSAMPLING/tabpfn/Results_UnderSampling_fold_{fold}_excluded_train_data_with_pred_proba.csv', index=False)

    # Calculate evaluation scores
    accuracy_scores.append(accuracy_score(y_val, y_pred))
    f1_scores.append(f1_score(y_val, y_pred))
    roc_auc_scores.append(roc_auc_score(y_val, y_pred_proba))
    precision_scores.append(precision_score(y_val, y_pred))
    recall_scores.append(recall_score(y_val, y_pred))

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)

    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc_scores[-1]:.2f})')

# Configure plot details
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/CV_5FOLD_UNDERSAMPLING/tabpfn/roc_curve_fold_tabpfn.png', dpi=300)
plt.show()

# %%
# Calculate mean values
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
mean_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
mean_sensitivity = sum(sensitivity_scores) / len(sensitivity_scores)
mean_specificity = sum(specificity_scores) / len(specificity_scores)

# Display evaluation results for each fold
print(f'Accuracy scores: {accuracy_scores}')
print(f'Mean accuracy: {mean_accuracy}')
print(f'F1 scores: {f1_scores}')
print(f'Mean F1 score: {mean_f1}')
print(f'ROC AUC scores: {roc_auc_scores}')
print(f'Mean ROC AUC score: {mean_roc_auc}')
print(f'Precision scores: {precision_scores}')
print(f'Mean Precision: {mean_precision}')
print(f'Recall scores: {recall_scores}')
print(f'Mean Recall: {mean_recall}')
print(f'Sensitivity scores: {sensitivity_scores}')
print(f'Mean Sensitivity: {mean_sensitivity}')
print(f'Specificity scores: {specificity_scores}')
print(f'Mean Specificity: {mean_specificity}')

# Create the CSV file
with open('analysis_outputs/CV_5FOLD_UNDERSAMPLING/tabpfn/evaluation_results_5fold_tabpfn.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['Metric', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean'])

    # Write each score
    writer.writerow(['Accuracy'] + accuracy_scores + [mean_accuracy])
    writer.writerow(['F1 Score'] + f1_scores + [mean_f1])
    writer.writerow(['ROC AUC'] + roc_auc_scores + [mean_roc_auc])
    writer.writerow(['Precision'] + precision_scores + [mean_precision])
    writer.writerow(['Recall'] + recall_scores + [mean_recall])
    writer.writerow(['Sensitivity'] + sensitivity_scores + [mean_sensitivity])
    writer.writerow(['Specificity'] + specificity_scores + [mean_specificity])

print('CSVファイルに評価結果を出力しました。')

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Hyperparameters updated on 20240717
# Logistic Regression
logistic_regression_params = {'C': 0.001, 'penalty': 'none', 'solver': 'saga', 'random_state': 0}

# Random Forest
random_forest_params = {'max_depth': 10, 'n_estimators': 100, 'random_state': 0}

# XGBoost
xgboost_params = {'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 50, 'random_state': 0}

# LightGBM
lightgbm_params = {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100, 'random_state': 0}

param_grids = {
    'Logistic Regression': logistic_regression_params,
    'Random Forest': random_forest_params,
    'XGBoost': xgboost_params,
    'LightGBM': lightgbm_params
}

# Initialize the model
models = {
    'Logistic Regression': LogisticRegression(**logistic_regression_params),
    'Random Forest': RandomForestClassifier(**random_forest_params),
    'XGBoost': XGBClassifier(**xgboost_params),
    'LightGBM': LGBMClassifier(**lightgbm_params)
}

# %%
# Set the directory where fold data are saved
output_dir = 'analysis_outputs'

# List for saving evaluation results
accuracy_scores = []
f1_scores = []
roc_auc_scores = []
precision_scores = []
recall_scores = []
sensitivity_scores = []
specificity_scores = []

# Plot settings for drawing ROC curves
plt.figure(figsize=(10, 8))

# Load data for each fold, then train and evaluate the model
for fold in range(5):
    # Load training and validation data
    train_data = pd.read_csv(f'{output_dir}/fold_{fold}_train_data_after_imputation.csv')
    val_data = pd.read_csv(f'{output_dir}/fold_{fold}_val_data_after_imputation.csv')

    # Split into features and target variable using only features selected by RFECV
    X_val = val_data[rfecv_features]
    y_val = val_data['Failure level (0 vs 1 to 3) (6M)']

    # Undersample the data
    train_data_rus = pd.read_csv(f'{output_dir}/fold_{fold}_undersampled_train_data.csv')
    X_rus = train_data_rus[rfecv_features]
    y_rus = train_data_rus['Failure level (0 vs 1 to 3) (6M)']

    # Train the model
    rf = models['Random Forest']
    rf.fit(X_rus, y_rus)

    # Save the model
    joblib_file = f"{output_dir}/CV_5FOLD_UNDERSAMPLING/rf/rf_model_fold_{fold}.pkl"
    joblib.dump(rf, joblib_file)
    print(f"Model for fold {fold} saved to {joblib_file}")

    # Output predicted probabilities for validation data to a CSV file
    y_pred = rf.predict(X_val)
    y_pred_proba = rf.predict_proba(X_val)[:, 1]
    val_data['pred_proba'] = y_pred_proba
    val_data.to_csv(f'{output_dir}/CV_5FOLD_UNDERSAMPLING/rf/Results_UnderSampling_fold_{fold}_val_data_with_pred_proba.csv', index=False)

    # Data excluded by undersampling
    train_data_excluded = pd.read_csv(f'{output_dir}/fold_{fold}_excluded_train_data.csv')
    X_train = train_data_excluded[rfecv_features]
    y_train = train_data_excluded['Failure level (0 vs 1 to 3) (6M)']

    # Calculate predicted probabilities for excluded data, merge them into train_data_excluded, and output to a CSV file
    y_pred_proba_excluded = rf.predict_proba(X_train)[:, 1]
    train_data_excluded['pred_proba'] = y_pred_proba_excluded
    train_data_excluded.to_csv(f'{output_dir}/CV_5FOLD_UNDERSAMPLING/rf/Results_UnderSampling_fold_{fold}_excluded_train_data_with_pred_proba.csv', index=False)

    # Calculate evaluation scores
    accuracy_scores.append(accuracy_score(y_val, y_pred))
    f1_scores.append(f1_score(y_val, y_pred))
    roc_auc_scores.append(roc_auc_score(y_val, y_pred_proba))
    precision_scores.append(precision_score(y_val, y_pred))
    recall_scores.append(recall_score(y_val, y_pred))

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)

    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc_scores[-1]:.2f})')

# Configure plot details
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save the plot at 300 dpi
plt.savefig('analysis_outputs/CV_5FOLD_UNDERSAMPLING/rf/roc_curve_fold_rf.png', dpi=300)
plt.show()

# %%
# Calculate mean values
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
mean_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
mean_sensitivity = sum(sensitivity_scores) / len(sensitivity_scores)
mean_specificity = sum(specificity_scores) / len(specificity_scores)

# Display evaluation results for each fold
print(f'Accuracy scores: {accuracy_scores}')
print(f'Mean accuracy: {mean_accuracy}')
print(f'F1 scores: {f1_scores}')
print(f'Mean F1 score: {mean_f1}')
print(f'ROC AUC scores: {roc_auc_scores}')
print(f'Mean ROC AUC score: {mean_roc_auc}')
print(f'Precision scores: {precision_scores}')
print(f'Mean Precision: {mean_precision}')
print(f'Recall scores: {recall_scores}')
print(f'Mean Recall: {mean_recall}')
print(f'Sensitivity scores: {sensitivity_scores}')
print(f'Mean Sensitivity: {mean_sensitivity}')
print(f'Specificity scores: {specificity_scores}')
print(f'Mean Specificity: {mean_specificity}')

# Create the CSV file
with open('analysis_outputs/CV_5FOLD_UNDERSAMPLING/rf/evaluation_results_5fold_rf.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['Metric', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean'])

    # Write each score
    writer.writerow(['Accuracy'] + accuracy_scores + [mean_accuracy])
    writer.writerow(['F1 Score'] + f1_scores + [mean_f1])
    writer.writerow(['ROC AUC'] + roc_auc_scores + [mean_roc_auc])
    writer.writerow(['Precision'] + precision_scores + [mean_precision])
    writer.writerow(['Recall'] + recall_scores + [mean_recall])
    writer.writerow(['Sensitivity'] + sensitivity_scores + [mean_sensitivity])
    writer.writerow(['Specificity'] + specificity_scores + [mean_specificity])

print('CSVファイルに評価結果を出力しました。')

# %%
# Set the directory where fold data are saved
output_dir = 'analysis_outputs'

# List for saving evaluation results
accuracy_scores = []
f1_scores = []
roc_auc_scores = []
precision_scores = []
recall_scores = []
sensitivity_scores = []
specificity_scores = []

# Plot settings for drawing ROC curves
plt.figure(figsize=(10, 8))

# Load data for each fold, then train and evaluate the model
for fold in range(5):
    # Load training and validation data
    train_data = pd.read_csv(f'{output_dir}/fold_{fold}_train_data_after_imputation.csv')
    val_data = pd.read_csv(f'{output_dir}/fold_{fold}_val_data_after_imputation.csv')

    # Split into features and target variable using only features selected by RFECV
    X_val = val_data[rfecv_features]
    y_val = val_data['Failure level (0 vs 1 to 3) (6M)']

    # Undersample the data
    train_data_rus = pd.read_csv(f'{output_dir}/fold_{fold}_undersampled_train_data.csv')
    X_rus = train_data_rus[rfecv_features]
    y_rus = train_data_rus['Failure level (0 vs 1 to 3) (6M)']

    # Train the model
    xgb = models['XGBoost']
    xgb.fit(X_rus, y_rus)

    # Save the model
    joblib_file = f"{output_dir}/CV_5FOLD_UNDERSAMPLING/xgb/xgb_model_fold_{fold}.pkl"
    joblib.dump(xgb, joblib_file)
    print(f"Model for fold {fold} saved to {joblib_file}")

    # Output predicted probabilities for validation data to a CSV file
    y_pred = xgb.predict(X_val)
    y_pred_proba = xgb.predict_proba(X_val)[:, 1]
    val_data['pred_proba'] = y_pred_proba
    val_data.to_csv(f'{output_dir}/CV_5FOLD_UNDERSAMPLING/xgb/Results_UnderSampling_fold_{fold}_val_data_with_pred_proba.csv', index=False)

    # Data excluded by undersampling
    train_data_excluded = pd.read_csv(f'{output_dir}/fold_{fold}_excluded_train_data.csv')
    X_train = train_data_excluded[rfecv_features]
    y_train = train_data_excluded['Failure level (0 vs 1 to 3) (6M)']

    # Calculate predicted probabilities for excluded data, merge them into train_data_excluded, and output to a CSV file
    y_pred_proba_excluded = xgb.predict_proba(X_train)[:, 1]
    train_data_excluded['pred_proba'] = y_pred_proba_excluded
    train_data_excluded.to_csv(f'{output_dir}/CV_5FOLD_UNDERSAMPLING/xgb/Results_UnderSampling_fold_{fold}_excluded_train_data_with_pred_proba.csv', index=False)

    # Calculate evaluation scores
    accuracy_scores.append(accuracy_score(y_val, y_pred))
    f1_scores.append(f1_score(y_val, y_pred))
    roc_auc_scores.append(roc_auc_score(y_val, y_pred_proba))
    precision_scores.append(precision_score(y_val, y_pred))
    recall_scores.append(recall_score(y_val, y_pred))

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)

    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc_scores[-1]:.2f})')

# Configure plot details
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/CV_5FOLD_UNDERSAMPLING/xgb/roc_curve_fold_xgb.png', dpi=300)
plt.show()

# %%
# Calculate mean values
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
mean_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
mean_sensitivity = sum(sensitivity_scores) / len(sensitivity_scores)
mean_specificity = sum(specificity_scores) / len(specificity_scores)

# Display evaluation results for each fold
print(f'Accuracy scores: {accuracy_scores}')
print(f'Mean accuracy: {mean_accuracy}')
print(f'F1 scores: {f1_scores}')
print(f'Mean F1 score: {mean_f1}')
print(f'ROC AUC scores: {roc_auc_scores}')
print(f'Mean ROC AUC score: {mean_roc_auc}')
print(f'Precision scores: {precision_scores}')
print(f'Mean Precision: {mean_precision}')
print(f'Recall scores: {recall_scores}')
print(f'Mean Recall: {mean_recall}')
print(f'Sensitivity scores: {sensitivity_scores}')
print(f'Mean Sensitivity: {mean_sensitivity}')
print(f'Specificity scores: {specificity_scores}')
print(f'Mean Specificity: {mean_specificity}')

# Create the CSV file
with open('analysis_outputs/CV_5FOLD_UNDERSAMPLING/xgb/evaluation_results_5fold_xgb.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['Metric', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean'])

    # Write each score
    writer.writerow(['Accuracy'] + accuracy_scores + [mean_accuracy])
    writer.writerow(['F1 Score'] + f1_scores + [mean_f1])
    writer.writerow(['ROC AUC'] + roc_auc_scores + [mean_roc_auc])
    writer.writerow(['Precision'] + precision_scores + [mean_precision])
    writer.writerow(['Recall'] + recall_scores + [mean_recall])
    writer.writerow(['Sensitivity'] + sensitivity_scores + [mean_sensitivity])
    writer.writerow(['Specificity'] + specificity_scores + [mean_specificity])

print('CSVファイルに評価結果を出力しました。')

# %%
# Set the directory where fold data are saved
output_dir = 'analysis_outputs'

# List for saving evaluation results
accuracy_scores = []
f1_scores = []
roc_auc_scores = []
precision_scores = []
recall_scores = []
sensitivity_scores = []
specificity_scores = []

# Plot settings for drawing ROC curves
plt.figure(figsize=(10, 8))

# Load data for each fold, then train and evaluate the model
for fold in range(5):
    # Load training and validation data
    train_data = pd.read_csv(f'{output_dir}/fold_{fold}_train_data_after_imputation.csv')
    val_data = pd.read_csv(f'{output_dir}/fold_{fold}_val_data_after_imputation.csv')

    # Split into features and target variable using only features selected by RFECV
    X_val = val_data[rfecv_features]
    y_val = val_data['Failure level (0 vs 1 to 3) (6M)']

    # Undersample the data
    train_data_rus = pd.read_csv(f'{output_dir}/fold_{fold}_undersampled_train_data.csv')
    X_rus = train_data_rus[rfecv_features]
    y_rus = train_data_rus['Failure level (0 vs 1 to 3) (6M)']

    # Train the model
    lgb = models['LightGBM']
    lgb.fit(X_rus, y_rus)

    # Save the model
    joblib_file = f"{output_dir}/CV_5FOLD_UNDERSAMPLING/lgb/lgb_model_fold_{fold}.pkl"
    joblib.dump(lgb, joblib_file)
    print(f"Model for fold {fold} saved to {joblib_file}")

    # Output predicted probabilities for validation data to a CSV file
    y_pred = lgb.predict(X_val)
    y_pred_proba = lgb.predict_proba(X_val)[:, 1]
    val_data['pred_proba'] = y_pred_proba
    val_data.to_csv(f'{output_dir}/CV_5FOLD_UNDERSAMPLING/lgb/Results_UnderSampling_fold_{fold}_val_data_with_pred_proba.csv', index=False)

    # Data excluded by undersampling
    train_data_excluded = pd.read_csv(f'{output_dir}/fold_{fold}_excluded_train_data.csv')
    X_train = train_data_excluded[rfecv_features]
    y_train = train_data_excluded['Failure level (0 vs 1 to 3) (6M)']

    # Calculate predicted probabilities for excluded data, merge them into train_data_excluded, and output to a CSV file
    y_pred_proba_excluded = lgb.predict_proba(X_train)[:, 1]
    train_data_excluded['pred_proba'] = y_pred_proba_excluded
    train_data_excluded.to_csv(f'{output_dir}/CV_5FOLD_UNDERSAMPLING/lgb/Results_UnderSampling_fold_{fold}_excluded_train_data_with_pred_proba.csv', index=False)

    # Calculate evaluation scores
    accuracy_scores.append(accuracy_score(y_val, y_pred))
    f1_scores.append(f1_score(y_val, y_pred))
    roc_auc_scores.append(roc_auc_score(y_val, y_pred_proba))
    precision_scores.append(precision_score(y_val, y_pred))
    recall_scores.append(recall_score(y_val, y_pred))

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)

    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc_scores[-1]:.2f})')

# Configure plot details
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/CV_5FOLD_UNDERSAMPLING/lgb/roc_curve_fold_lgb.png', dpi=300)
plt.show()

# %%
# Calculate mean values
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
mean_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
mean_sensitivity = sum(sensitivity_scores) / len(sensitivity_scores)
mean_specificity = sum(specificity_scores) / len(specificity_scores)

# Display evaluation results for each fold
print(f'Accuracy scores: {accuracy_scores}')
print(f'Mean accuracy: {mean_accuracy}')
print(f'F1 scores: {f1_scores}')
print(f'Mean F1 score: {mean_f1}')
print(f'ROC AUC scores: {roc_auc_scores}')
print(f'Mean ROC AUC score: {mean_roc_auc}')
print(f'Precision scores: {precision_scores}')
print(f'Mean Precision: {mean_precision}')
print(f'Recall scores: {recall_scores}')
print(f'Mean Recall: {mean_recall}')
print(f'Sensitivity scores: {sensitivity_scores}')
print(f'Mean Sensitivity: {mean_sensitivity}')
print(f'Specificity scores: {specificity_scores}')
print(f'Mean Specificity: {mean_specificity}')

# Create the CSV file
with open('analysis_outputs/CV_5FOLD_UNDERSAMPLING/lgb/evaluation_results_5fold_lgb.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['Metric', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean'])

    # Write each score
    writer.writerow(['Accuracy'] + accuracy_scores + [mean_accuracy])
    writer.writerow(['F1 Score'] + f1_scores + [mean_f1])
    writer.writerow(['ROC AUC'] + roc_auc_scores + [mean_roc_auc])
    writer.writerow(['Precision'] + precision_scores + [mean_precision])
    writer.writerow(['Recall'] + recall_scores + [mean_recall])
    writer.writerow(['Sensitivity'] + sensitivity_scores + [mean_sensitivity])
    writer.writerow(['Specificity'] + specificity_scores + [mean_specificity])

print('CSVファイルに評価結果を出力しました。')

# %%
# Set the directory where fold data are saved
output_dir = 'analysis_outputs'

# List for saving evaluation results
accuracy_scores = []
f1_scores = []
roc_auc_scores = []
precision_scores = []
recall_scores = []
sensitivity_scores = []
specificity_scores = []

# Plot settings for drawing ROC curves
plt.figure(figsize=(10, 8))

# Load data for each fold, then train and evaluate the model
for fold in range(5):
    # Load training and validation data
    train_data = pd.read_csv(f'{output_dir}/fold_{fold}_train_data_after_imputation.csv')
    val_data = pd.read_csv(f'{output_dir}/fold_{fold}_val_data_after_imputation.csv')

    # Split into features and target variable using only features selected by RFECV
    X_val = val_data[rfecv_features]
    y_val = val_data['Failure level (0 vs 1 to 3) (6M)']

    # Undersample the data
    train_data_rus = pd.read_csv(f'{output_dir}/fold_{fold}_undersampled_train_data.csv')
    X_rus = train_data_rus[rfecv_features]
    y_rus = train_data_rus['Failure level (0 vs 1 to 3) (6M)']

    # Train the model
    LR = models['Logistic Regression']
    LR.fit(X_rus, y_rus)

    # Save the model
    joblib_file = f"{output_dir}/CV_5FOLD_UNDERSAMPLING/LR/LR_model_fold_{fold}.pkl"
    joblib.dump(LR, joblib_file)
    print(f"Model for fold {fold} saved to {joblib_file}")

    # Output predicted probabilities for validation data to a CSV file
    y_pred = LR.predict(X_val)
    y_pred_proba = LR.predict_proba(X_val)[:, 1]
    val_data['pred_proba'] = y_pred_proba
    val_data.to_csv(f'{output_dir}/CV_5FOLD_UNDERSAMPLING/LR/Results_UnderSampling_fold_{fold}_val_data_with_pred_proba.csv', index=False)

    # Data excluded by undersampling
    train_data_excluded = pd.read_csv(f'{output_dir}/fold_{fold}_excluded_train_data.csv')
    X_train = train_data_excluded[rfecv_features]
    y_train = train_data_excluded['Failure level (0 vs 1 to 3) (6M)']

    # Calculate predicted probabilities for excluded data, merge them into train_data_excluded, and output to a CSV file
    y_pred_proba_excluded = LR.predict_proba(X_train)[:, 1]
    train_data_excluded['pred_proba'] = y_pred_proba_excluded
    train_data_excluded.to_csv(f'{output_dir}/CV_5FOLD_UNDERSAMPLING/LR/Results_UnderSampling_fold_{fold}_excluded_train_data_with_pred_proba.csv', index=False)

    # Calculate evaluation scores
    accuracy_scores.append(accuracy_score(y_val, y_pred))
    f1_scores.append(f1_score(y_val, y_pred))
    roc_auc_scores.append(roc_auc_score(y_val, y_pred_proba))
    precision_scores.append(precision_score(y_val, y_pred))
    recall_scores.append(recall_score(y_val, y_pred))

    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specificity)

    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {roc_auc_scores[-1]:.2f})')

# Configure plot details
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/CV_5FOLD_UNDERSAMPLING/LR/roc_curve_fold_LR.png', dpi=300)
plt.show()

# %%
# Calculate mean values
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
mean_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
mean_sensitivity = sum(sensitivity_scores) / len(sensitivity_scores)
mean_specificity = sum(specificity_scores) / len(specificity_scores)

# Display evaluation results for each fold
print(f'Accuracy scores: {accuracy_scores}')
print(f'Mean accuracy: {mean_accuracy}')
print(f'F1 scores: {f1_scores}')
print(f'Mean F1 score: {mean_f1}')
print(f'ROC AUC scores: {roc_auc_scores}')
print(f'Mean ROC AUC score: {mean_roc_auc}')
print(f'Precision scores: {precision_scores}')
print(f'Mean Precision: {mean_precision}')
print(f'Recall scores: {recall_scores}')
print(f'Mean Recall: {mean_recall}')
print(f'Sensitivity scores: {sensitivity_scores}')
print(f'Mean Sensitivity: {mean_sensitivity}')
print(f'Specificity scores: {specificity_scores}')
print(f'Mean Specificity: {mean_specificity}')

# Create the CSV file
with open('analysis_outputs/CV_5FOLD_UNDERSAMPLING/LR/evaluation_results_5fold_LR.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['Metric', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean'])

    # Write each score
    writer.writerow(['Accuracy'] + accuracy_scores + [mean_accuracy])
    writer.writerow(['F1 Score'] + f1_scores + [mean_f1])
    writer.writerow(['ROC AUC'] + roc_auc_scores + [mean_roc_auc])
    writer.writerow(['Precision'] + precision_scores + [mean_precision])
    writer.writerow(['Recall'] + recall_scores + [mean_recall])
    writer.writerow(['Sensitivity'] + sensitivity_scores + [mean_sensitivity])
    writer.writerow(['Specificity'] + specificity_scores + [mean_specificity])

print('CSVファイルに評価結果を出力しました。')

# %%


