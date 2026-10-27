import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('insurance.csv')
numeric_columns = ['age', 'bmi','children', 'charges']
# print(df.columns)
# EDA

# print(df.shape)
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# for col in numeric_columns:
#     plt.figure(figsize=(6,4))
#     sns.histplot(df[col], kde= True , bins=20)
#     plt.show()

# sns.countplot(x = df['children'])
# plt.show()
# sns.countplot(x = df['sex'])
# plt.show()
# sns.countplot(x = df['smoker'])
# plt.show()

# for col in numeric_columns:
#     plt.figure(figsize=(6,4))
#     sns.boxplot(df[col])
#     plt.show()

# DATA CLEANING and PREPROSESSING

df_cleaned = df.copy()

# print(df_cleaned.shape)
# df_cleaned.drop_duplicates(inplace=True)
# print(df_cleaned.shape)

# print(df_cleaned['sex'].value_counts())

#  label encoding

df_cleaned['sex'] = df_cleaned['sex'].map({'male':0 , 'female':1})
# print(df_cleaned.head())
df_cleaned['smoker'] = df_cleaned['smoker'].map({'no':0 , 'yes':1})
# print(df_cleaned.head())
df_cleaned.rename(columns=
                  {
                      'sex':'is_female',
                      'smoker':'is_smoker'
                  },inplace=True)

df_cleaned = pd.get_dummies(df_cleaned , columns=['region'],drop_first=True)
df_cleaned = df_cleaned.astype(int)
# print(df_cleaned.head())


#  FEATURE ENGINEERING AND EXTRACTION

df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels=['Underweight','Normal','Overweight','Obese']
)

df_cleaned = pd.get_dummies(df_cleaned , columns=['bmi_category'],drop_first=True)
df_cleaned = df_cleaned.astype(int)

# print(df_cleaned.head())

#  FEATURE SCALING

from sklearn.preprocessing import StandardScaler
cols=['age','bmi','children']
scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])

# print(df_cleaned.head())

# feature correlation

from scipy.stats import pearsonr

selected_features = [
    'age', 'bmi', 'children', 'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight',
    'bmi_category_Obese'
]

correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}

correlation_df = pd.DataFrame(
    list(correlations.items()),
    columns=['Feature', 'Pearson Correlation']
)

correlation_df.sort_values(
    by='Pearson Correlation',
    ascending=False
)

# print(correlation_df)


#  chi square test

cat_features = [
    'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

from scipy.stats import chi2_contingency

alpha = 0.05

df_cleaned['charges_bin'] = pd.qcut(
    df_cleaned['charges'],
    q=4,
    labels=False
)

chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)

    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'

    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_results).T

chi2_df = chi2_df.sort_values(by='p_value')

# print(chi2_df)

final_df = df_cleaned[['age', 'is_female', 'bmi', 'children', 'is_smoker', 'charges','region_southeast','bmi_category_Obese','region_northwest']]

# print(final_df)


#  model training and testing

from sklearn.model_selection import train_test_split

X = final_df.drop('charges' , axis=1)  #input features 
y = final_df['charges']   # output feature

#  split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#  train the model

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train , y_train)

y_pred = model.predict(X_test)

# print(y_pred)

#  performance evaluation

from sklearn.metrics import r2_score

r2 = r2_score(y_test,y_pred)
print(r2)

n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = adjusted_r2 = 1 - ((1- r2) * (n-1) / (n - p - 1))

print(adjusted_r2)