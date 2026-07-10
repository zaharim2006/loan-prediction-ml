import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

print("Hammasi tayyor!")
df = pd.read_csv('train.csv')

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

# ============ DATA CLEANING ============

# Kategorik ustunlar - eng ko'p uchraydigan qiymat (mode) bilan to'ldiramiz
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Sonli ustun - median bilan to'ldiramiz (chunki chetga chiqib ketgan qiymatlar - outliers - bor)
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())

# Tekshirish - endi hech qanday bo'sh joy qolmasligi kerak
print("\n=== Tozalashdan keyin ===")
print(df.isnull().sum())

# # ============ EDA (Exploratory Data Analysis) ============

# 1. Loan_Status taqsimoti (nechta ha, nechta yo'q)
print("\n=== Loan_Status taqsimoti ===")
print(df['Loan_Status'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='Loan_Status', data=df)
plt.title("Kredit berilgan/berilmagan mijozlar soni")
plt.show()

# 2. Gender bo'yicha Loan_Status
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Loan_Status', data=df)
plt.title("Jins bo'yicha kredit natijasi")
plt.show()

# 3. Married bo'yicha Loan_Status
plt.figure(figsize=(6,4))
sns.countplot(x='Married', hue='Loan_Status', data=df)
plt.title("Turmush holati bo'yicha kredit natijasi")
plt.show()

# 4. Education bo'yicha Loan_Status
plt.figure(figsize=(6,4))
sns.countplot(x='Education', hue='Loan_Status', data=df)
plt.title("Ta'lim darajasi bo'yicha kredit natijasi")
plt.show()

# 5. ENG MUHIMI: Credit_History bo'yicha Loan_Status
plt.figure(figsize=(6,4))
sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
plt.title("Kredit tarixi bo'yicha kredit natijasi")
plt.show()

# 6. ApplicantIncome taqsimoti
plt.figure(figsize=(6,4))
sns.histplot(df['ApplicantIncome'], bins=30, kde=True)
plt.title("Mijoz daromadining taqsimoti")
plt.show()

# 7. Property_Area bo'yicha Loan_Status
plt.figure(figsize=(6,4))
sns.countplot(x='Property_Area', hue='Loan_Status', data=df)
plt.title("Mulk hududi bo'yicha kredit natijasi")
plt.show()

# ============ FEATURE ENGINEERING ============

# 1. Umumiy daromad - yangi ustun yaratamiz
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# 2. Log transformatsiya - cho'zilgan taqsimotni normallashtirish uchun
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'] = np.log(df['LoanAmount'])

# 3. Kerak bo'lmagan ustunlarni olib tashlaymiz
df = df.drop(['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome'], axis=1)

# 4. Kategorik ustunlarni raqamga aylantiramiz (Label Encoding)
from sklearn.preprocessing import LabelEncoder

cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 5. Natijani tekshiramiz
print("\n=== Feature Engineering dan keyin ===")
print(df.head())
print(df.dtypes)

# ============ MODEL TRAINING ============

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. X (features) va y (target) ni ajratamiz
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# 2. Train/Test ga bo'lamiz (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train hajmi: {X_train.shape}")
print(f"Test hajmi: {X_test.shape}")

# 3. Modellarni tayyorlaymiz
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

# 4. Har bir modelni o'qitamiz va baholaymiz
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # ============ HYPERPARAMETER TUNING ============

from sklearn.model_selection import GridSearchCV

# 1. Random Forest uchun parametrlar
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("\n=== Random Forest Tuning ===")
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("Eng yaxshi parametrlar:", rf_grid.best_params_)
print("Eng yaxshi F1-score (CV):", rf_grid.best_score_)

rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Test F1-score: {f1_score(y_test, y_pred_rf):.4f}")

# 2. XGBoost uchun parametrlar
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

print("\n=== XGBoost Tuning ===")
xgb_grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), xgb_params, cv=5, scoring='f1', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

print("Eng yaxshi parametrlar:", xgb_grid.best_params_)
print("Eng yaxshi F1-score (CV):", xgb_grid.best_score_)

xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Test F1-score: {f1_score(y_test, y_pred_xgb):.4f}")


# ============ YAKUNIY BOSQICH ============

import joblib

# 1. Eng yaxshi modelni saqlaymiz
joblib.dump(xgb_best, 'best_model.pkl')
print("\nModel 'best_model.pkl' fayliga saqlandi!")

# 2. Feature Importance - qaysi ustun eng muhim
importances = xgb_best.feature_importances_
feature_names = X.columns

feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

print("\n=== Feature Importance ===")
print(feat_imp_df)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title("Qaysi omillar kredit qaroriga eng ko'p ta'sir qiladi")
plt.tight_layout()
plt.show()