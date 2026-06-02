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

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
from imblearn.under_sampling import RandomUnderSampler
import csv

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier()
}

# Hyperparameters updated on 20240717
# Logistic Regression
logistic_regression_params = {'C': 0.001, 'penalty': 'none', 'solver': 'saga', 'random_state': 123}

# Random Forest
random_forest_params = {'max_depth': 5, 'n_estimators': 200, 'random_state': 123}

# XGBoost
xgboost_params = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200, 'random_state': 123}

# LightGBM
lightgbm_params = {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 50, 'random_state': 123}

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
# Load the imputed training dataset
df_train = pd.read_csv('analysis_outputs/train_data_after_imputation.csv')
df_test = pd.read_csv('analysis_outputs/test_data_after_imputation.csv')

#REFCV_selected_columns
# Load the list from a CSV file
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
# Create the TabPFN model
tabpfn = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)
tabpfn.fit(X_tr, y_tr, overwrite_warning = True)

# Set the directory where fold data are saved
output_dir = 'analysis_outputs'

# Save the model
joblib_file = f'{output_dir}/FINAL_ORIGINAL/tabpfn/tabpfn_model_ORIGINAL_FINAL.pkl'
joblib.dump(tabpfn, joblib_file)

print(f"Model saved to {joblib_file}")

# %%
# Set the directory where fold data are saved
output_dir = 'analysis_outputs'

# Predict
y_pred = tabpfn.predict(X_test_final)
y_pred_proba = tabpfn.predict_proba(X_test_final)[:, 1]
y_pred_proba_tabpfn = y_pred_proba.copy()

# Output predicted probabilities for test data to a CSV file
df_test['pred_proba'] = y_pred_proba
df_test.to_csv(f'{output_dir}/FINAL_ORIGINAL/tabpfn/Results_ORIGINAL_with_pred_proba.csv', index=False)

# %%
import csv

output_dir = 'analysis_outputs'

# Calculate metrics.
acc = accuracy_score(y_test_final, y_pred)
auc_score = roc_auc_score(y_test_final, y_pred_proba)
recall = recall_score(y_test_final, y_pred)
precision = precision_score(y_test_final, y_pred)
f1 = f1_score(y_test_final, y_pred)
kappa = cohen_kappa_score(y_test_final, y_pred)
mcc = matthews_corrcoef(y_test_final, y_pred)

# Calculate the confusion matrix
tabpfn_cm = confusion_matrix(y_test_final, y_pred)
TN, FP, FN, TP = tabpfn_cm.ravel()

# Calculate sensitivity (true positive rate)
sensitivity = TP / (TP + FN)

# Calculate specificity (true negative rate)
specificity = TN / (TN + FP)

# Calculate positive predictive value
positive_predictive_value = TP / (TP + FP)

# Calculate negative predictive value
negative_predictive_value = TN / (TN + FN)

# Display results.
print(f'Accuracy: {acc}')
print(f'AUC: {auc_score}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')
print(f'Kappa: {kappa}')
print(f'MCC: {mcc}')
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Positive Predictive Value: {positive_predictive_value}')
print(f'Negative Predictive Value: {negative_predictive_value}')

# Write metrics to a CSV file.
with open(f'{output_dir}/FINAL_ORIGINAL/tabpfn/metrics_tabpfn.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Accuracy", acc])
    writer.writerow(["AUC", auc_score])
    writer.writerow(["Recall", recall])
    writer.writerow(["Precision", precision])
    writer.writerow(["F1 Score", f1])
    writer.writerow(["Kappa", kappa])
    writer.writerow(["MCC", mcc])
    writer.writerow(["Sensitivity", sensitivity])
    writer.writerow(["Specificity", specificity])
    writer.writerow(["Positive Predictive Value", positive_predictive_value])
    writer.writerow(["Negative Predictive Value", negative_predictive_value])

# %%
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate the ROC curve using the true test labels
fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/tabpfn/roc_curve_tabpfn_final.png', dpi=300)
plt.show()

# %%
# Visualize the confusion matrix

import seaborn as sns
from matplotlib import pyplot

tabpfn_cm = confusion_matrix(y_test_final, y_pred)

f = pyplot.figure(figsize=(8,8))

sns.heatmap(tabpfn_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')

labels = ['0', '1']
pyplot.xticks([0.5,1.5], labels, fontsize=16, fontweight='heavy')
pyplot.yticks([0.5,1.5], labels, fontsize=16, fontweight='heavy', va='center')

pyplot.xlabel('Predicted', fontsize=22, fontweight='heavy', labelpad=16)
pyplot.ylabel('Truth', fontsize=22, fontweight='heavy', labelpad=16)

pyplot.tick_params(axis="y",direction="out", pad=10)
pyplot.tick_params(axis="x",direction="out", pad=10)
pyplot.title('', x = -0.095, y = 1.005, fontsize = 75, pad = 20)

pyplot.subplots_adjust(left=0.20, right=0.85, bottom=0.20, top=0.80)
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/tabpfn/confusion_matrix_tabpfn_final.png', dpi=300)
pyplot.show()

# %%
import matplotlib.pyplot as plt

# Predicted probabilities for samples that are actually positive
positive_pred_probs = y_pred_proba[y_test_final == 1]

# Predicted probabilities for samples that are actually negative
negative_pred_probs = y_pred_proba[y_test_final == 0]

# Histogram of positive samples
plt.hist(positive_pred_probs, bins=50, alpha=0.5, label='Positive', color='g')

# Histogram of negative samples
plt.hist(negative_pred_probs, bins=50, alpha=0.5, label='Negative', color='r')

plt.xlabel('Predicted Probability of Positive Class')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.legend(loc='upper right')
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/tabpfn/prob_tabpfn_final.png', dpi=300)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Calculate sensitivity and specificity while varying the predicted-probability threshold from 0 to 1
thresholds = np.linspace(0, 1, 100)
sensitivities = []
specificities = []

for threshold in thresholds:
    # Generate predicted labels
    y_test_pred = y_pred_proba > threshold
    tn, fp, fn, tp = confusion_matrix(y_test_final, y_test_pred).ravel()

    # Sensitivity (true positive rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivities.append(sensitivity)

    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(specificity)

# Find the threshold that minimizes the difference between sensitivity and specificity
differences = np.abs(np.array(sensitivities) - np.array(specificities))
min_diff_index = np.argmin(differences)
intersection_threshold = thresholds[min_diff_index]
intersection_sensitivity = sensitivities[min_diff_index]
intersection_specificity = specificities[min_diff_index]

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(thresholds, sensitivities, label='Sensitivity', color='red')
plt.plot(thresholds, specificities, label='Specificity', color='blue')
plt.scatter(intersection_threshold, intersection_sensitivity, color='green', label='Intersection Point')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Sensitivity and Specificity vs. Threshold')
plt.legend()
plt.grid(True)
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/tabpfn/sensitivity_specificity_plot_tabpfn_final.png', dpi=300)
plt.show()

(intersection_threshold, intersection_sensitivity, intersection_specificity)

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Generate the threshold list
thresholds = np.arange(0.1, 1.0, 0.1)

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Threshold', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])

for threshold in thresholds:
    # Generate predicted labels using the threshold
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    # Calculate the confusion matrix
    TN, FP, FN, TP = confusion_matrix(y_test_final, y_pred_threshold).ravel()

    # Calculate sensitivity, specificity, positive predictive value, and negative predictive value
    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    ppv = TP / (TP + FP) if TP + FP > 0 else 0
    npv = TN / (TN + FN) if TN + FN > 0 else 0

    # Add results to the DataFrame
    results_row = pd.DataFrame({'Threshold': [threshold], 'Sensitivity': [sensitivity], 'Specificity': [specificity], 'PPV': [ppv], 'NPV': [npv]})
    results_df = pd.concat([results_df, results_row], ignore_index=True)

# Save results to a CSV file
csv_file_path = 'analysis_outputs/FINAL_ORIGINAL/tabpfn/threshold_metrics_tabpfn.csv'
results_df.to_csv(csv_file_path, index=False)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate the threshold list
thresholds = np.arange(0.1, 1.0, 0.1)

# Determine the number of subplots
n_cols = 3
n_rows = (len(thresholds) + n_cols - 1) // n_cols

# Set the plot size
plt.figure(figsize=(n_cols * 6, n_rows * 6))

for i, threshold in enumerate(thresholds, start=1):
    # Generate predicted labels using the threshold
    y_pred_labels = (y_pred_proba >= threshold).astype(int)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test_final, y_pred_labels)

    # Create subplots
    plt.subplot(n_rows, n_cols, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')
    plt.title(f'Threshold: {threshold:.1f}', fontsize=22)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Truth', fontsize=18)

plt.tight_layout()
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/tabpfn/threshold_confusion_matrix_tabpfn.png', dpi=300)
plt.show()

# %%
import pandas as pd

# Create a copy of the test dataset
test_data_with_probabilities = X_test_final.copy()

# Add columns for true labels and predicted probabilities
test_data_with_probabilities['True Label'] = y_test_final
test_data_with_probabilities['Predicted Probability'] = y_pred_proba

# %%
import matplotlib.pyplot as plt
import seaborn as sns

output_file_path = 'analysis_outputs/FINAL_ORIGINAL/tabpfn/'

# Copy the data
data_with_probabilities = test_data_with_probabilities.copy()

# Get feature names as a list, excluding 'True Label' and 'Predicted Probability'
feature_columns = data_with_probabilities.columns[:-2]

# Group features in sets of five, draw scatter plots, and save them
for i in range(0, len(feature_columns), 5):
    plt.figure(figsize=(20, 4))
    for j, feature in enumerate(feature_columns[i:i+5], start=1):
        plt.subplot(1, 5, j)
        # Plot points for negative labels
        sns.scatterplot(x='Predicted Probability', y=feature, data=data_with_probabilities[data_with_probabilities['True Label'] == 0], color='gray', alpha=0.3, label='True Label 0')
        # Plot points for positive labels
        sns.scatterplot(x='Predicted Probability', y=feature, data=data_with_probabilities[data_with_probabilities['True Label'] == 1], color='red', alpha=0.8, label='True Label 1')
        plt.title(feature)
    plt.tight_layout()
    # Save the figure
    plt.savefig(f'{output_file_path}/scatter_plot_group_{i//5+1}_tabpfn.png')
    plt.show()

# %%
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

# Create the calibration plot
prob_true, prob_pred = calibration_curve(y_test_final, y_pred_proba, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration plot')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.ylabel('Fraction of positives')
plt.xlabel('Mean predicted value')
plt.title('Calibration Plot')
plt.legend(loc='best')
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/tabpfn/calibration_Plot_tabpfn.png', dpi=300)
plt.show()

# %%
tabpfn_explainer = shap.Explainer(tabpfn.predict, X_test_final)
shap_values = tabpfn_explainer(X_test_final)

# %%
import shap
import matplotlib.pyplot as plt

# Plot SHAP values and test data
shap.summary_plot(shap_values, X_test_final, show=False)

# Save the figure to a file (DPI 300)
plt.savefig('analysis_outputs/FINAL_ORIGINAL/tabpfn/shap_summary_plot_tabpfn.png', bbox_inches='tight', dpi=300)
plt.close()

# %%
# Train the model
rf = models['Random Forest']
rf.fit(X_tr, y_tr)

output_dir = 'analysis_outputs'

# Save the model
joblib_file = f'{output_dir}/FINAL_ORIGINAL/rf/rf_model_ORIGINAL_FINAL.pkl'
joblib.dump(rf, joblib_file)

print(f"Model saved to {joblib_file}")

# Predict
y_pred = rf.predict(X_test_final)
y_pred_proba = rf.predict_proba(X_test_final)[:, 1]
y_pred_proba_rf = y_pred_proba.copy()

# Output predicted probabilities for test data to a CSV file
df_test['pred_proba'] = y_pred_proba
df_test.to_csv(f'{output_dir}/FINAL_ORIGINAL/rf/Results_ORIGINAL_with_pred_proba.csv', index=False)

# %%
import csv

output_dir = 'analysis_outputs'

# Calculate metrics.
acc = accuracy_score(y_test_final, y_pred)
auc_score = roc_auc_score(y_test_final, y_pred_proba)
recall = recall_score(y_test_final, y_pred)
precision = precision_score(y_test_final, y_pred)
f1 = f1_score(y_test_final, y_pred)
kappa = cohen_kappa_score(y_test_final, y_pred)
mcc = matthews_corrcoef(y_test_final, y_pred)

# Calculate the confusion matrix
rf_cm = confusion_matrix(y_test_final, y_pred)
TN, FP, FN, TP = rf_cm.ravel()

# Calculate sensitivity (true positive rate)
sensitivity = TP / (TP + FN)

# Calculate specificity (true negative rate)
specificity = TN / (TN + FP)

# Calculate positive predictive value
positive_predictive_value = TP / (TP + FP)

# Calculate negative predictive value
negative_predictive_value = TN / (TN + FN)

# Display results.
print(f'Accuracy: {acc}')
print(f'AUC: {auc_score}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')
print(f'Kappa: {kappa}')
print(f'MCC: {mcc}')
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Positive Predictive Value: {positive_predictive_value}')
print(f'Negative Predictive Value: {negative_predictive_value}')

# Write metrics to a CSV file.
with open(f'{output_dir}/FINAL_ORIGINAL/rf/metrics_rf.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Accuracy", acc])
    writer.writerow(["AUC", auc_score])
    writer.writerow(["Recall", recall])
    writer.writerow(["Precision", precision])
    writer.writerow(["F1 Score", f1])
    writer.writerow(["Kappa", kappa])
    writer.writerow(["MCC", mcc])
    writer.writerow(["Sensitivity", sensitivity])
    writer.writerow(["Specificity", specificity])
    writer.writerow(["Positive Predictive Value", positive_predictive_value])
    writer.writerow(["Negative Predictive Value", negative_predictive_value])

# %%
# Get feature importances
importance = rf.feature_importances_

# Convert feature importances to a DataFrame
importance_df = pd.DataFrame({
    'Feature': X_tr.columns,
    'Importance': importance
})

# Sort feature importances in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances as a bar chart
plt.figure(figsize=(14, 10))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/rf/feature_importance_ORIGINAL_rf.png', dpi=300)
plt.show()

# Save feature importances to a CSV file
importance_df.to_csv('analysis_outputs/FINAL_ORIGINAL/rf/feature_importance_ORIGINAL_rf.csv', index=False)

# %%
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate the ROC curve using the true test labels
fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/rf/roc_curve_rf_final.png', dpi=300)
plt.show()

# %%
# Visualize the confusion matrix
import seaborn as sns
from matplotlib import pyplot

rf_cm = confusion_matrix(y_test_final, y_pred)

f = pyplot.figure(figsize=(8,8))

sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')

labels = ['0', '1']
pyplot.xticks([0.5,1.5], labels, fontsize=16, fontweight='heavy')
pyplot.yticks([0.5,1.5], labels, fontsize=16, fontweight='heavy', va='center')

pyplot.xlabel('Predicted', fontsize=22, fontweight='heavy', labelpad=16)
pyplot.ylabel('Truth', fontsize=22, fontweight='heavy', labelpad=16)

pyplot.tick_params(axis="y",direction="out", pad=10)
pyplot.tick_params(axis="x",direction="out", pad=10)
pyplot.title('', x = -0.095, y = 1.005, fontsize = 75, pad = 20)

pyplot.subplots_adjust(left=0.20, right=0.85, bottom=0.20, top=0.80)
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/rf/confusion_matrix_rf_final.png', dpi=300)
pyplot.show()

# %%
import matplotlib.pyplot as plt

# Predicted probabilities for samples that are actually positive
positive_pred_probs = y_pred_proba[y_test_final == 1]

# Predicted probabilities for samples that are actually negative
negative_pred_probs = y_pred_proba[y_test_final == 0]

# Histogram of positive samples
plt.hist(positive_pred_probs, bins=50, alpha=0.5, label='Positive', color='g')

# Histogram of negative samples
plt.hist(negative_pred_probs, bins=50, alpha=0.5, label='Negative', color='r')

plt.xlabel('Predicted Probability of Positive Class')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.legend(loc='upper right')
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/rf/prob_rf_final.png', dpi=300)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Calculate sensitivity and specificity while varying the predicted-probability threshold from 0 to 1
thresholds = np.linspace(0, 1, 100)
sensitivities = []
specificities = []

for threshold in thresholds:
    # Generate predicted labels
    y_test_pred = y_pred_proba > threshold
    tn, fp, fn, tp = confusion_matrix(y_test_final, y_test_pred).ravel()

    # Sensitivity (true positive rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivities.append(sensitivity)

    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(specificity)

# Find the threshold that minimizes the difference between sensitivity and specificity
differences = np.abs(np.array(sensitivities) - np.array(specificities))
min_diff_index = np.argmin(differences)
intersection_threshold = thresholds[min_diff_index]
intersection_sensitivity = sensitivities[min_diff_index]
intersection_specificity = specificities[min_diff_index]

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(thresholds, sensitivities, label='Sensitivity', color='red')
plt.plot(thresholds, specificities, label='Specificity', color='blue')
plt.scatter(intersection_threshold, intersection_sensitivity, color='green', label='Intersection Point')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Sensitivity and Specificity vs. Threshold')
plt.legend()
plt.grid(True)
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/rf/sensitivity_specificity_plot_rf_final.png', dpi=300)
plt.show()

(intersection_threshold, intersection_sensitivity, intersection_specificity)

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Generate the threshold list
thresholds = np.arange(0.1, 1.0, 0.1)

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Threshold', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])

for threshold in thresholds:
    # Generate predicted labels using the threshold
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    # Calculate the confusion matrix
    TN, FP, FN, TP = confusion_matrix(y_test_final, y_pred_threshold).ravel()

    # Calculate sensitivity, specificity, positive predictive value, and negative predictive value
    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    ppv = TP / (TP + FP) if TP + FP > 0 else 0
    npv = TN / (TN + FN) if TN + FN > 0 else 0

    # Add results to the DataFrame
    results_row = pd.DataFrame({'Threshold': [threshold], 'Sensitivity': [sensitivity], 'Specificity': [specificity], 'PPV': [ppv], 'NPV': [npv]})
    results_df = pd.concat([results_df, results_row], ignore_index=True)

# Save results to a CSV file
csv_file_path = 'analysis_outputs/FINAL_ORIGINAL/rf/threshold_metrics_rf.csv'
results_df.to_csv(csv_file_path, index=False)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate the threshold list
thresholds = np.arange(0.1, 1.0, 0.1)

# Determine the number of subplots
n_cols = 3
n_rows = (len(thresholds) + n_cols - 1) // n_cols

# Set the plot size
plt.figure(figsize=(n_cols * 6, n_rows * 6))

for i, threshold in enumerate(thresholds, start=1):
    # Generate predicted labels using the threshold
    y_pred_labels = (y_pred_proba >= threshold).astype(int)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test_final, y_pred_labels)

    # Create subplots
    plt.subplot(n_rows, n_cols, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')
    plt.title(f'Threshold: {threshold:.1f}', fontsize=22)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Truth', fontsize=18)

plt.tight_layout()
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/rf/threshold_confusion_matrix_rf.png', dpi=300)
plt.show()

# %%
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

# Create the calibration plot
prob_true, prob_pred = calibration_curve(y_test_final, y_pred_proba, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration plot')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.ylabel('Fraction of positives')
plt.xlabel('Mean predicted value')
plt.title('Calibration Plot')
plt.legend(loc='best')
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/rf/calibration_Plot_rf.png', dpi=300)
plt.show()

# %%
# Calculate SHAP values
explainer = shap.Explainer(rf.predict, X_test_final)
shap_values = explainer(X_test_final)

# %%
import shap
import matplotlib.pyplot as plt

# Plot SHAP values and test data
shap.summary_plot(shap_values, X_test_final, show=False)

# Save the figure to a file (DPI 300)
plt.savefig('analysis_outputs/FINAL_ORIGINAL/rf/shap_summary_plot_rf.png', bbox_inches='tight', dpi=300)
plt.close()

# %%
# Train the model
xgb = models['XGBoost']
xgb.fit(X_tr, y_tr)

output_dir = 'analysis_outputs'

# Save the model
joblib_file = f'{output_dir}/FINAL_ORIGINAL/xgb/xgb_model_ORIGINAL_FINAL.pkl'
joblib.dump(xgb, joblib_file)

print(f"Model saved to {joblib_file}")

# Predict
y_pred = xgb.predict(X_test_final)
y_pred_proba = xgb.predict_proba(X_test_final)[:, 1]
y_pred_proba_xgb = y_pred_proba.copy()

# Output predicted probabilities for test data to a CSV file
df_test['pred_proba'] = y_pred_proba
df_test.to_csv(f'{output_dir}/FINAL_ORIGINAL/xgb/Results_ORIGINAL_with_pred_proba.csv', index=False)

# %%
import csv

output_dir = 'analysis_outputs'

# Calculate metrics.
acc = accuracy_score(y_test_final, y_pred)
auc_score = roc_auc_score(y_test_final, y_pred_proba)
recall = recall_score(y_test_final, y_pred)
precision = precision_score(y_test_final, y_pred)
f1 = f1_score(y_test_final, y_pred)
kappa = cohen_kappa_score(y_test_final, y_pred)
mcc = matthews_corrcoef(y_test_final, y_pred)

# Calculate the confusion matrix
xgb_cm = confusion_matrix(y_test_final, y_pred)
TN, FP, FN, TP = xgb_cm.ravel()

# Calculate sensitivity (true positive rate)
sensitivity = TP / (TP + FN)

# Calculate specificity (true negative rate)
specificity = TN / (TN + FP)

# Calculate positive predictive value
positive_predictive_value = TP / (TP + FP)

# Calculate negative predictive value
negative_predictive_value = TN / (TN + FN)

# Display results.
print(f'Accuracy: {acc}')
print(f'AUC: {auc_score}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')
print(f'Kappa: {kappa}')
print(f'MCC: {mcc}')
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Positive Predictive Value: {positive_predictive_value}')
print(f'Negative Predictive Value: {negative_predictive_value}')

# Write metrics to a CSV file.
with open(f'{output_dir}/FINAL_ORIGINAL/xgb/metrics_xgb.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Accuracy", acc])
    writer.writerow(["AUC", auc_score])
    writer.writerow(["Recall", recall])
    writer.writerow(["Precision", precision])
    writer.writerow(["F1 Score", f1])
    writer.writerow(["Kappa", kappa])
    writer.writerow(["MCC", mcc])
    writer.writerow(["Sensitivity", sensitivity])
    writer.writerow(["Specificity", specificity])
    writer.writerow(["Positive Predictive Value", positive_predictive_value])
    writer.writerow(["Negative Predictive Value", negative_predictive_value])

# %%
# Get feature importances
importance = xgb.feature_importances_

# Convert feature importances to a DataFrame
importance_df = pd.DataFrame({
    'Feature': X_tr.columns,
    'Importance': importance
})

# Sort feature importances in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances as a bar chart
plt.figure(figsize=(14, 10))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/xgb/feature_importance_ORIGINAL_xgb.png', dpi=300)
plt.show()

# Save feature importances to a CSV file
importance_df.to_csv('analysis_outputs/FINAL_ORIGINAL/xgb/feature_importance_ORIGINAL_xgb.csv', index=False)

# %%
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate the ROC curve using the true test labels
fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/xgb/roc_curve_xgb_final.png', dpi=300)
plt.show()

# %%
# Visualize the confusion matrix
import seaborn as sns
from matplotlib import pyplot

xgb_cm = confusion_matrix(y_test_final, y_pred)

f = pyplot.figure(figsize=(8,8))

sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')

labels = ['0', '1']
pyplot.xticks([0.5,1.5], labels, fontsize=16, fontweight='heavy')
pyplot.yticks([0.5,1.5], labels, fontsize=16, fontweight='heavy', va='center')

pyplot.xlabel('Predicted', fontsize=22, fontweight='heavy', labelpad=16)
pyplot.ylabel('Truth', fontsize=22, fontweight='heavy', labelpad=16)

pyplot.tick_params(axis="y",direction="out", pad=10)
pyplot.tick_params(axis="x",direction="out", pad=10)
pyplot.title('', x = -0.095, y = 1.005, fontsize = 75, pad = 20)

pyplot.subplots_adjust(left=0.20, right=0.85, bottom=0.20, top=0.80)
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/xgb/confusion_matrix_xgb_final.png', dpi=300)
pyplot.show()

# %%
import matplotlib.pyplot as plt

# Predicted probabilities for samples that are actually positive
positive_pred_probs = y_pred_proba[y_test_final == 1]

# Predicted probabilities for samples that are actually negative
negative_pred_probs = y_pred_proba[y_test_final == 0]

# Histogram of positive samples
plt.hist(positive_pred_probs, bins=50, alpha=0.5, label='Positive', color='g')

# Histogram of negative samples
plt.hist(negative_pred_probs, bins=50, alpha=0.5, label='Negative', color='r')

plt.xlabel('Predicted Probability of Positive Class')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.legend(loc='upper right')
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/xgb/prob_xgb_final.png', dpi=300)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Calculate sensitivity and specificity while varying the predicted-probability threshold from 0 to 1
thresholds = np.linspace(0, 1, 100)
sensitivities = []
specificities = []

for threshold in thresholds:
    # Generate predicted labels
    y_test_pred = y_pred_proba > threshold
    tn, fp, fn, tp = confusion_matrix(y_test_final, y_test_pred).ravel()

    # Sensitivity (true positive rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivities.append(sensitivity)

    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(specificity)

# Find the threshold that minimizes the difference between sensitivity and specificity
differences = np.abs(np.array(sensitivities) - np.array(specificities))
min_diff_index = np.argmin(differences)
intersection_threshold = thresholds[min_diff_index]
intersection_sensitivity = sensitivities[min_diff_index]
intersection_specificity = specificities[min_diff_index]

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(thresholds, sensitivities, label='Sensitivity', color='red')
plt.plot(thresholds, specificities, label='Specificity', color='blue')
plt.scatter(intersection_threshold, intersection_sensitivity, color='green', label='Intersection Point')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Sensitivity and Specificity vs. Threshold')
plt.legend()
plt.grid(True)
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/xgb/sensitivity_specificity_plot_xgb_final.png', dpi=300)
plt.show()

(intersection_threshold, intersection_sensitivity, intersection_specificity)

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Generate the threshold list
thresholds = np.arange(0.1, 1.0, 0.1)

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Threshold', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])

for threshold in thresholds:
    # Generate predicted labels using the threshold
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    # Calculate the confusion matrix
    TN, FP, FN, TP = confusion_matrix(y_test_final, y_pred_threshold).ravel()

    # Calculate sensitivity, specificity, positive predictive value, and negative predictive value
    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    ppv = TP / (TP + FP) if TP + FP > 0 else 0
    npv = TN / (TN + FN) if TN + FN > 0 else 0

    # Add results to the DataFrame
    results_row = pd.DataFrame({'Threshold': [threshold], 'Sensitivity': [sensitivity], 'Specificity': [specificity], 'PPV': [ppv], 'NPV': [npv]})
    results_df = pd.concat([results_df, results_row], ignore_index=True)

# Save results to a CSV file
csv_file_path = 'analysis_outputs/FINAL_ORIGINAL/xgb/threshold_metrics_xgb.csv'
results_df.to_csv(csv_file_path, index=False)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate the threshold list
thresholds = np.arange(0.1, 1.0, 0.1)

# Determine the number of subplots
n_cols = 3
n_rows = (len(thresholds) + n_cols - 1) // n_cols

# Set the plot size
plt.figure(figsize=(n_cols * 6, n_rows * 6))

for i, threshold in enumerate(thresholds, start=1):
    # Generate predicted labels using the threshold
    y_pred_labels = (y_pred_proba >= threshold).astype(int)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test_final, y_pred_labels)

    # Create subplots
    plt.subplot(n_rows, n_cols, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')
    plt.title(f'Threshold: {threshold:.1f}', fontsize=22)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Truth', fontsize=18)

plt.tight_layout()
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/xgb/threshold_confusion_matrix_xgb.png', dpi=300)
plt.show()

# %%
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

# Create the calibration plot
prob_true, prob_pred = calibration_curve(y_test_final, y_pred_proba, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration plot')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.ylabel('Fraction of positives')
plt.xlabel('Mean predicted value')
plt.title('Calibration Plot')
plt.legend(loc='best')
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/xgb/calibration_Plot_xgb.png', dpi=300)
plt.show()

# %%
# Interpretability of Random Forest using SHAP
import shap

# Calculate SHAP values
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test_final)

# %%
import shap
import matplotlib.pyplot as plt

# Plot SHAP values and test data
shap.summary_plot(shap_values, X_test_final, show=False)

# Save the figure to a file (DPI 300)
plt.savefig('analysis_outputs/FINAL_ORIGINAL/xgb/shap_summary_plot_xgb.png', bbox_inches='tight', dpi=300)
plt.close()

# %%
# Train the model
lgb = models['LightGBM']
lgb.fit(X_tr, y_tr)

output_dir = 'analysis_outputs'

# Save the model
joblib_file = f'{output_dir}/FINAL_ORIGINAL/lgb/lgb_model_ORIGINAL_FINAL.pkl'
joblib.dump(lgb, joblib_file)

print(f"Model saved to {joblib_file}")

# Predict
y_pred = lgb.predict(X_test_final)
y_pred_proba = lgb.predict_proba(X_test_final)[:, 1]
y_pred_proba_lgb = y_pred_proba.copy()

# Output predicted probabilities for test data to a CSV file
df_test['pred_proba'] = y_pred_proba
df_test.to_csv(f'{output_dir}/FINAL_ORIGINAL/lgb/Results_ORIGINAL_with_pred_proba.csv', index=False)

# %%
import csv

output_dir = 'analysis_outputs'

# Calculate metrics.
acc = accuracy_score(y_test_final, y_pred)
auc_score = roc_auc_score(y_test_final, y_pred_proba)
recall = recall_score(y_test_final, y_pred)
precision = precision_score(y_test_final, y_pred)
f1 = f1_score(y_test_final, y_pred)
kappa = cohen_kappa_score(y_test_final, y_pred)
mcc = matthews_corrcoef(y_test_final, y_pred)

# Calculate the confusion matrix
lgb_cm = confusion_matrix(y_test_final, y_pred)
TN, FP, FN, TP = lgb_cm.ravel()

# Calculate sensitivity (true positive rate)
sensitivity = TP / (TP + FN)

# Calculate specificity (true negative rate)
specificity = TN / (TN + FP)

# Calculate positive predictive value
positive_predictive_value = TP / (TP + FP)

# Calculate negative predictive value
negative_predictive_value = TN / (TN + FN)

# Display results.
print(f'Accuracy: {acc}')
print(f'AUC: {auc_score}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')
print(f'Kappa: {kappa}')
print(f'MCC: {mcc}')
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Positive Predictive Value: {positive_predictive_value}')
print(f'Negative Predictive Value: {negative_predictive_value}')

# Write metrics to a CSV file.
with open(f'{output_dir}/FINAL_ORIGINAL/lgb/metrics_lgb.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Accuracy", acc])
    writer.writerow(["AUC", auc_score])
    writer.writerow(["Recall", recall])
    writer.writerow(["Precision", precision])
    writer.writerow(["F1 Score", f1])
    writer.writerow(["Kappa", kappa])
    writer.writerow(["MCC", mcc])
    writer.writerow(["Sensitivity", sensitivity])
    writer.writerow(["Specificity", specificity])
    writer.writerow(["Positive Predictive Value", positive_predictive_value])
    writer.writerow(["Negative Predictive Value", negative_predictive_value])

# %%
# Get feature importances
importance = lgb.feature_importances_

# Convert feature importances to a DataFrame
importance_df = pd.DataFrame({
    'Feature': X_tr.columns,
    'Importance': importance
})

# Sort feature importances in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances as a bar chart
plt.figure(figsize=(14, 10))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/lgb/feature_importance_ORIGINAL_lgb.png', dpi=300)
plt.show()

# Save feature importances to a CSV file
importance_df.to_csv('analysis_outputs/FINAL_ORIGINAL/lgb/feature_importance_ORIGINAL_lgb.csv', index=False)

# %%
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate the ROC curve using the true test labels
fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/lgb/roc_curve_lgb_final.png', dpi=300)
plt.show()

# %%
# Visualize the confusion matrix
import seaborn as sns
from matplotlib import pyplot

lgb_cm = confusion_matrix(y_test_final, y_pred)

f = pyplot.figure(figsize=(8,8))

sns.heatmap(lgb_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')

labels = ['0', '1']
pyplot.xticks([0.5,1.5], labels, fontsize=16, fontweight='heavy')
pyplot.yticks([0.5,1.5], labels, fontsize=16, fontweight='heavy', va='center')

pyplot.xlabel('Predicted', fontsize=22, fontweight='heavy', labelpad=16)
pyplot.ylabel('Truth', fontsize=22, fontweight='heavy', labelpad=16)

pyplot.tick_params(axis="y",direction="out", pad=10)
pyplot.tick_params(axis="x",direction="out", pad=10)
pyplot.title('', x = -0.095, y = 1.005, fontsize = 75, pad = 20)

pyplot.subplots_adjust(left=0.20, right=0.85, bottom=0.20, top=0.80)
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/lgb/confusion_matrix_lgb_final.png', dpi=300)
pyplot.show()

# %%
import matplotlib.pyplot as plt

# Predicted probabilities for samples that are actually positive
positive_pred_probs = y_pred_proba[y_test_final == 1]

# Predicted probabilities for samples that are actually negative
negative_pred_probs = y_pred_proba[y_test_final == 0]

# Histogram of positive samples
plt.hist(positive_pred_probs, bins=50, alpha=0.5, label='Positive', color='g')

# Histogram of negative samples
plt.hist(negative_pred_probs, bins=50, alpha=0.5, label='Negative', color='r')

plt.xlabel('Predicted Probability of Positive Class')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.legend(loc='upper right')
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/lgb/prob_lgb_final.png', dpi=300)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Calculate sensitivity and specificity while varying the predicted-probability threshold from 0 to 1
thresholds = np.linspace(0, 1, 100)
sensitivities = []
specificities = []

for threshold in thresholds:
    # Generate predicted labels
    y_test_pred = y_pred_proba > threshold
    tn, fp, fn, tp = confusion_matrix(y_test_final, y_test_pred).ravel()

    # Sensitivity (true positive rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivities.append(sensitivity)

    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(specificity)

# Find the threshold that minimizes the difference between sensitivity and specificity
differences = np.abs(np.array(sensitivities) - np.array(specificities))
min_diff_index = np.argmin(differences)
intersection_threshold = thresholds[min_diff_index]
intersection_sensitivity = sensitivities[min_diff_index]
intersection_specificity = specificities[min_diff_index]

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(thresholds, sensitivities, label='Sensitivity', color='red')
plt.plot(thresholds, specificities, label='Specificity', color='blue')
plt.scatter(intersection_threshold, intersection_sensitivity, color='green', label='Intersection Point')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Sensitivity and Specificity vs. Threshold')
plt.legend()
plt.grid(True)
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/lgb/sensitivity_specificity_plot_lgb_final.png', dpi=300)
plt.show()

(intersection_threshold, intersection_sensitivity, intersection_specificity)

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Generate the threshold list
thresholds = np.arange(0.1, 1.0, 0.1)

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Threshold', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])

for threshold in thresholds:
    # Generate predicted labels using the threshold
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    # Calculate the confusion matrix
    TN, FP, FN, TP = confusion_matrix(y_test_final, y_pred_threshold).ravel()

    # Calculate sensitivity, specificity, positive predictive value, and negative predictive value
    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    ppv = TP / (TP + FP) if TP + FP > 0 else 0
    npv = TN / (TN + FN) if TN + FN > 0 else 0

    # Add results to the DataFrame
    results_row = pd.DataFrame({'Threshold': [threshold], 'Sensitivity': [sensitivity], 'Specificity': [specificity], 'PPV': [ppv], 'NPV': [npv]})
    results_df = pd.concat([results_df, results_row], ignore_index=True)

# Save results to a CSV file
csv_file_path = 'analysis_outputs/FINAL_ORIGINAL/lgb/threshold_metrics_lgb.csv'
results_df.to_csv(csv_file_path, index=False)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate the threshold list
thresholds = np.arange(0.1, 1.0, 0.1)

# Determine the number of subplots
n_cols = 3
n_rows = (len(thresholds) + n_cols - 1) // n_cols

# Set the plot size
plt.figure(figsize=(n_cols * 6, n_rows * 6))

for i, threshold in enumerate(thresholds, start=1):
    # Generate predicted labels using the threshold
    y_pred_labels = (y_pred_proba >= threshold).astype(int)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test_final, y_pred_labels)

    # Create subplots
    plt.subplot(n_rows, n_cols, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')
    plt.title(f'Threshold: {threshold:.1f}', fontsize=22)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Truth', fontsize=18)

plt.tight_layout()
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/lgb/threshold_confusion_matrix_lgb.png', dpi=300)
plt.show()

# %%
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

# Create the calibration plot
prob_true, prob_pred = calibration_curve(y_test_final, y_pred_proba, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration plot')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.ylabel('Fraction of positives')
plt.xlabel('Mean predicted value')
plt.title('Calibration Plot')
plt.legend(loc='best')
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/lgb/calibration_Plot_lgb.png', dpi=300)
plt.show()

# %%
# Calculate SHAP values
explainer = shap.TreeExplainer(lgb)
shap_values = explainer.shap_values(X_test_final)

# %%
import shap
import matplotlib.pyplot as plt

# Plot SHAP values and test data
shap.summary_plot(shap_values, X_test_final, show=False)

# Save the figure to a file (DPI 300)
plt.savefig('analysis_outputs/FINAL_ORIGINAL/lgb/shap_summary_plot_lgb.png', bbox_inches='tight', dpi=300)
plt.close()

# %%
# Train the model
LR = models['Logistic Regression']
LR.fit(X_tr, y_tr)

# Predict
y_pred = LR.predict(X_test_final)
y_pred_proba = LR.predict_proba(X_test_final)[:, 1]
y_pred_proba_LR = y_pred_proba.copy()

# Save the model
joblib_file = f'{output_dir}/FINAL_ORIGINAL/LR/LR_model_ORIGINAL_FINAL.pkl'
joblib.dump(LR, joblib_file)

print(f"Model saved to {joblib_file}")

# Output predicted probabilities for test data to a CSV file
df_test['pred_proba'] = y_pred_proba
df_test.to_csv(f'{output_dir}/FINAL_ORIGINAL/LR/Results_ORIGINAL_with_pred_proba.csv', index=False)

# %%
import csv

output_dir = 'analysis_outputs'

# Calculate metrics.
acc = accuracy_score(y_test_final, y_pred)
auc_score = roc_auc_score(y_test_final, y_pred_proba)
recall = recall_score(y_test_final, y_pred)
precision = precision_score(y_test_final, y_pred)
f1 = f1_score(y_test_final, y_pred)
kappa = cohen_kappa_score(y_test_final, y_pred)
mcc = matthews_corrcoef(y_test_final, y_pred)

# Calculate the confusion matrix
LR_cm = confusion_matrix(y_test_final, y_pred)
TN, FP, FN, TP = LR_cm.ravel()

# Calculate sensitivity (true positive rate)
sensitivity = TP / (TP + FN)

# Calculate specificity (true negative rate)
specificity = TN / (TN + FP)

# Calculate positive predictive value
positive_predictive_value = TP / (TP + FP)

# Calculate negative predictive value
negative_predictive_value = TN / (TN + FN)

# Display results.
print(f'Accuracy: {acc}')
print(f'AUC: {auc_score}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')
print(f'Kappa: {kappa}')
print(f'MCC: {mcc}')
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'Positive Predictive Value: {positive_predictive_value}')
print(f'Negative Predictive Value: {negative_predictive_value}')

# Write metrics to a CSV file.
with open(f'{output_dir}/FINAL_ORIGINAL/LR/metrics_LR.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Accuracy", acc])
    writer.writerow(["AUC", auc_score])
    writer.writerow(["Recall", recall])
    writer.writerow(["Precision", precision])
    writer.writerow(["F1 Score", f1])
    writer.writerow(["Kappa", kappa])
    writer.writerow(["MCC", mcc])
    writer.writerow(["Sensitivity", sensitivity])
    writer.writerow(["Specificity", specificity])
    writer.writerow(["Positive Predictive Value", positive_predictive_value])
    writer.writerow(["Negative Predictive Value", negative_predictive_value])

# %%
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calculate the ROC curve using the true test labels
fpr, tpr, _ = roc_curve(y_test_final, y_pred_proba)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/LR/roc_curve_LR_final.png', dpi=300)
plt.show()

# %%
# Visualize the confusion matrix
import seaborn as sns
from matplotlib import pyplot

LR_cm = confusion_matrix(y_test_final, y_pred)

f = pyplot.figure(figsize=(8,8))

sns.heatmap(LR_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')

labels = ['0', '1']
pyplot.xticks([0.5,1.5], labels, fontsize=16, fontweight='heavy')
pyplot.yticks([0.5,1.5], labels, fontsize=16, fontweight='heavy', va='center')

pyplot.xlabel('Predicted', fontsize=22, fontweight='heavy', labelpad=16)
pyplot.ylabel('Truth', fontsize=22, fontweight='heavy', labelpad=16)

pyplot.tick_params(axis="y",direction="out", pad=10)
pyplot.tick_params(axis="x",direction="out", pad=10)
pyplot.title('', x = -0.095, y = 1.005, fontsize = 75, pad = 20)

pyplot.subplots_adjust(left=0.20, right=0.85, bottom=0.20, top=0.80)
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/LR/confusion_matrix_LR_final.png', dpi=300)
pyplot.show()

# %%
import matplotlib.pyplot as plt

# Predicted probabilities for samples that are actually positive
positive_pred_probs = y_pred_proba[y_test_final == 1]

# Predicted probabilities for samples that are actually negative
negative_pred_probs = y_pred_proba[y_test_final == 0]

# Histogram of positive samples
plt.hist(positive_pred_probs, bins=50, alpha=0.5, label='Positive', color='g')

# Histogram of negative samples
plt.hist(negative_pred_probs, bins=50, alpha=0.5, label='Negative', color='r')

plt.xlabel('Predicted Probability of Positive Class')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.legend(loc='upper right')
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/LR/prob_LR_final.png', dpi=300)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Calculate sensitivity and specificity while varying the predicted-probability threshold from 0 to 1
thresholds = np.linspace(0, 1, 100)
sensitivities = []
specificities = []

for threshold in thresholds:
    # Generate predicted labels
    y_test_pred = y_pred_proba > threshold
    tn, fp, fn, tp = confusion_matrix(y_test_final, y_test_pred).ravel()

    # Sensitivity (true positive rate)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivities.append(sensitivity)

    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(specificity)

# Find the threshold that minimizes the difference between sensitivity and specificity
differences = np.abs(np.array(sensitivities) - np.array(specificities))
min_diff_index = np.argmin(differences)
intersection_threshold = thresholds[min_diff_index]
intersection_sensitivity = sensitivities[min_diff_index]
intersection_specificity = specificities[min_diff_index]

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(thresholds, sensitivities, label='Sensitivity', color='red')
plt.plot(thresholds, specificities, label='Specificity', color='blue')
plt.scatter(intersection_threshold, intersection_sensitivity, color='green', label='Intersection Point')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Sensitivity and Specificity vs. Threshold')
plt.legend()
plt.grid(True)
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/LR/sensitivity_specificity_plot_LR_final.png', dpi=300)
plt.show()

(intersection_threshold, intersection_sensitivity, intersection_specificity)

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Generate the threshold list
thresholds = np.arange(0.1, 1.0, 0.1)

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Threshold', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])

for threshold in thresholds:
    # Generate predicted labels using the threshold
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    # Calculate the confusion matrix
    TN, FP, FN, TP = confusion_matrix(y_test_final, y_pred_threshold).ravel()

    # Calculate sensitivity, specificity, positive predictive value, and negative predictive value
    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    ppv = TP / (TP + FP) if TP + FP > 0 else 0
    npv = TN / (TN + FN) if TN + FN > 0 else 0

    # Add results to the DataFrame
    results_row = pd.DataFrame({'Threshold': [threshold], 'Sensitivity': [sensitivity], 'Specificity': [specificity], 'PPV': [ppv], 'NPV': [npv]})
    results_df = pd.concat([results_df, results_row], ignore_index=True)

# Save results to a CSV file
csv_file_path = 'analysis_outputs/FINAL_ORIGINAL/LR/threshold_metrics_LR.csv'
results_df.to_csv(csv_file_path, index=False)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Generate the threshold list
thresholds = np.arange(0.1, 1.0, 0.1)

# Determine the number of subplots
n_cols = 3
n_rows = (len(thresholds) + n_cols - 1) // n_cols

# Set the plot size
plt.figure(figsize=(n_cols * 6, n_rows * 6))

for i, threshold in enumerate(thresholds, start=1):
    # Generate predicted labels using the threshold
    y_pred_labels = (y_pred_proba >= threshold).astype(int)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test_final, y_pred_labels)

    # Create subplots
    plt.subplot(n_rows, n_cols, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')
    plt.title(f'Threshold: {threshold:.1f}', fontsize=22)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('Truth', fontsize=18)

plt.tight_layout()
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/LR/threshold_confusion_matrix_LR.png', dpi=300)
plt.show()

# %%
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV

# Create the calibration plot
prob_true, prob_pred = calibration_curve(y_test_final, y_pred_proba, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration plot')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.ylabel('Fraction of positives')
plt.xlabel('Mean predicted value')
plt.title('Calibration Plot')
plt.legend(loc='best')
# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/LR/calibration_Plot_LR.png', dpi=300)
plt.show()

# %%
# Get model coefficients
coefficients = LR.coef_[0]
intercept = LR.intercept_

# Convert coefficients and feature names to a DataFrame
coef_df = pd.DataFrame({
    'Feature': X_tr.columns,
    'Coefficient': coefficients
})

# Add the intercept to the DataFrame
intercept_df = pd.DataFrame({
    'Feature': ['Intercept'],
    'Coefficient': intercept
})

# Add the intercept to the coefficient DataFrame
coef_df = pd.concat([intercept_df, coef_df], ignore_index=True)

# Output the coefficient DataFrame to a CSV file
coef_df.to_csv('analysis_outputs/FINAL_ORIGINAL/LR/Model_Coefficients.csv', index=False)

# %%
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# True test labels
y_test_final = df_test['Failure level (0 vs 1 to 3) (6M)']

# Calculate ROC curves for each model
fpr1, tpr1, _ = roc_curve(y_test_final, y_pred_proba_tabpfn)
fpr2, tpr2, _ = roc_curve(y_test_final, y_pred_proba_rf)
fpr3, tpr3, _ = roc_curve(y_test_final, y_pred_proba_xgb)
fpr4, tpr4, _ = roc_curve(y_test_final, y_pred_proba_lgb)
fpr5, tpr5, _ = roc_curve(y_test_final, y_pred_proba_LR)

# Calculate AUC for each model
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)
roc_auc4 = auc(fpr4, tpr4)
roc_auc5 = auc(fpr5, tpr5)

# Plot the ROC curve
plt.figure()

plt.plot(fpr1, tpr1, color='blue', lw=1, label='TabPFN (AUROC = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='green', lw=1, label='Random Forest (AUROC = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='red', lw=1, label='XGBoost (AUROC = %0.2f)' % roc_auc3)
plt.plot(fpr4, tpr4, color='purple', lw=1, label='LightGBM (AUROC = %0.2f)' % roc_auc4)
plt.plot(fpr5, tpr5, color='darkorange', lw=1, label='Logistic Regression (AUROC = %0.2f)' % roc_auc5)

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Save the plot at 300 dpi
plt.savefig('analysis_outputs/FINAL_ORIGINAL/roc_curve_all_models_ORIGINAL.png', dpi=300)
plt.show()

# %%


