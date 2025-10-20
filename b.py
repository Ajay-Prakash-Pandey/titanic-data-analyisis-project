# titanic_advanced_model.py
# Advanced Titanic ML Pipeline — EDA + Feature Engineering + Model Comparison + Submission

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -------------------------------------------------------------------
# 1️⃣ LOAD DATA
# -------------------------------------------------------------------
print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

df = pd.concat([train, test], sort=False)
print("Data combined. Shape:", df.shape)

# -------------------------------------------------------------------
# 2️⃣ FEATURE ENGINEERING
# -------------------------------------------------------------------
print("Engineering features...")

# Extract title from Name
df['Title'] = df['Name'].apply(lambda x: re.search(' ([A-Za-z]+)\.', x).group(1) if pd.notnull(x) else 'None')
df['Title'] = df['Title'].replace(
    ['Mlle', 'Ms', 'Lady', 'Countess', 'Mme', 'Capt', 'Col', 'Don', 
     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
df['Title'] = df['Title'].map(title_map).fillna(0)

# Cabin Deck
df['CabinDeck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')

# Family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# IsAlone feature
df['IsAlone'] = 1
df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Fare bins
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)

# Age bins
df['AgeBin'] = pd.cut(df['Age'].astype(int), 5, labels=False)

# Encode categorical variables
label = LabelEncoder()
for col in ['Sex', 'Embarked', 'CabinDeck']:
    df[col] = label.fit_transform(df[col].astype(str))

# -------------------------------------------------------------------
# 3️⃣ PREPARE FINAL DATASETS
# -------------------------------------------------------------------
train_clean = df[df['Survived'].notnull()].copy()
test_clean = df[df['Survived'].isnull()].copy()

features = [
    'Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
    'FamilySize', 'IsAlone', 'Title', 'CabinDeck', 'FareBin', 'AgeBin'
]

X = train_clean[features]
y = train_clean['Survived'].astype(int)
X_test = test_clean[features]

# -------------------------------------------------------------------
# 4️⃣ MODEL TRAINING & EVALUATION
# -------------------------------------------------------------------
print("\nTraining models and comparing performance...")

models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42)
}

results = {}
for name, model in models.items():
    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    results[name] = cv_score
    print(f"{name} CV Accuracy: {cv_score:.4f}")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\nBest model selected: {best_model_name} ({results[best_model_name]:.4f})")

# Train best model on full data
best_model.fit(X, y)

# -------------------------------------------------------------------
# 5️⃣ PREDICT AND GENERATE SUBMISSION
# -------------------------------------------------------------------
print("\nGenerating final predictions...")
test_clean['Survived'] = best_model.predict(X_test).astype(int)
submission = test_clean[['PassengerId', 'Survived']]
submission.to_csv("submissionn.csv", index=False)

print("\nsubmission.csv created successfully!")
print(f"Best Model: {best_model_name}, CV Accuracy: {results[best_model_name]:.4f}")
print("Upload submission.csv to Kaggle Titanic competition page.")
print("\nSample predictions:")
print(submission.head(10))
