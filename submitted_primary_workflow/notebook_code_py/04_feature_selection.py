# %%
!pip install japanize_matplotlib

# %%
# Load the imputed training dataset
df_train = pd.read_csv('analysis_outputs/train_data_after_imputation.csv')
df_test = pd.read_csv('analysis_outputs/test_data_after_imputation.csv')

# %%
df_train.shape

# %%
# Load the imputed training dataset
df_train = pd.read_csv('train_data_after_imputation.csv')
df_test = pd.read_csv('test_data_after_imputation.csv')

# %%
list = df_train.columns.tolist()
print(list)

# %%
# Import libraries
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
# Check cells with missing values
missing_values = df_train.isnull().sum()
missing_values[missing_values > 0]

# %%
df_train.shape

# %%
X = df_train.drop(['ID', 'Failure level (0 vs 1 to 3) (6M)'], axis=1)
y = df_train['Failure level (0 vs 1 to 3) (6M)']

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

# %%
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

columns = X.columns

# Using AUROC scoring instead of accuracy
clf_rf = RandomForestClassifier(random_state=42)
rfecv = RFECV(estimator=clf_rf,
              step=1,
              cv=5,
              scoring='roc_auc')  # 5-fold cross-validation
rfecv = rfecv.fit(X, y)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', columns[rfecv.support_])

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Train the random forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Get feature importances
feature_importances = clf.feature_importances_

# Convert feature names and their importances to a DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': columns,
    'Importance': feature_importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

top_features_list = feature_importance_df.head(30)['Feature'].tolist()
# Display the list
print(top_features_list)

# Display top features as a bar plot
# Set the color uniformly to blue
plt.figure(figsize=(10, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(30), color='navy')
plt.title('Feature Importance')
plt.show()

# %%
#REFCV_selected_columns
rfecv_features = ['患者情報__性別', '術前所見__黄斑剥離', '術前所見_黄斑剥離期間_日', '術前所見_矯正視力_矯正視力',
       '術前所見_眼圧_眼圧', '術前所見__脈絡膜剥離', '術前所見_裂孔数_個', 'Hypotony_5mmHg',
       'V22a眼軸長(26mm以上)', '術前所見__主病名1_PVDに伴う弁状裂孔による裂孔原性網膜剥離',
       '術前所見__主病名1_その他の裂孔原性網膜剥離', '術前所見__主病名1_内眼手術後(白内障硝子体)の裂孔原性網膜剥離',
       '術前所見__主病名1_強度近視に伴う黄斑円孔網膜剥離', '術前所見__主病名1_萎縮円孔による裂孔原性網膜剥離',
       '術前所見_最大裂孔大きさ_度_0-30', '術前所見_最大裂孔大きさ_度_30-60', '術前所見_PVR_N/B/C_B',
       '術前所見_PVR_N/B/C_C', '術前所見_PVR_N/B/C_N', '術前所見__最大裂孔位置_上耳側',
       '術前所見__最大裂孔位置_上鼻側', '術前所見__最大裂孔位置_下耳側', '術前所見__最大裂孔位置_下鼻側',
       '術前所見__最大裂孔位置_後極', '術前所見_裂孔形態_種別_円孔', '術前所見_裂孔形態_種別_裂孔',
       '術前所見_裂孔形態_種別_黄斑円孔', '術前所見_網膜剥離範囲_現象_1', '術前所見_網膜剥離範囲_現象_2',
       '術前所見_網膜剥離範囲_現象_3', '術前所見_網膜剥離範囲_現象_4', '眼手術歴_網膜剥離を除く網膜硝子体手術',
       '眼手術歴_白内障手術', '術前所見__水晶体_有水晶体眼', '術前所見__水晶体_IOL', '初回手術時年齢']

# %%
import csv
import pandas as pd
import os

# Specify the path
output_directory = 'analysis_outputs/'
csv_filename = 'rfecv_features.csv'
full_path = os.path.join(output_directory, csv_filename)

# Write the list to a CSV file
with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for feature in rfecv_features:
        writer.writerow([feature])

print(f"特徴量リストを {full_path} に書き込みました。")

# Load the list from a CSV file
df = pd.read_csv(full_path, header=None, encoding='utf-8')
loaded_features = df[0].tolist()

print("CSVから読み込んだ特徴量リスト:")
print(loaded_features)

# Check whether the original list and loaded list are identical
if rfecv_features == loaded_features:
    print("元のリストと読み込んだリストは一致しています。")
else:
    print("警告: 元のリストと読み込んだリストが一致していません。")

# %%
# Load the list from a CSV file
# df = pd.read_csv(full_path, header=None, encoding='utf-8')
# rfecv_features = df[0].tolist()

# print("Feature list loaded from CSV:")
# print(rfecv_features)
