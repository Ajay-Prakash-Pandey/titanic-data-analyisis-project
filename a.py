# titanic_full_pipeline.py
# Complete Titanic competition workflow: EDA + Feature Engineering + Model + Submission
# Works cleanly on Windows (no emojis)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------
# 1️⃣ LOAD DATA
# --------------------------------------------------------------------
print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Data loaded successfully!")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print("\nTrain columns:", list(train.columns))

# --------------------------------------------------------------------
# 2️⃣ BASIC EDA
# --------------------------------------------------------------------
print("\nStarting Exploratory Data Analysis (EDA)...")

print("\n--- Missing Values ---")
print(train.isnull().sum())

# Numerical summaries
print("\n--- Numerical Features Summary ---")
print(train.describe())

# Plot survival distribution
sns.countplot(x='Survived', data=train)
plt.title("Survival Distribution")
plt.show()

# Gender vs Survival
sns.barplot(x='Sex', y='Survived', data=train)
plt.title("Survival Rate by Gender")
plt.show()

# Pclass vs Survival
sns.barplot(x='Pclass', y='Survived', data=train)
plt.title("Survival Rate by Passenger Class")
plt.show()

# Age distribution
train['Age'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# --------------------------------------------------------------------
# 3️⃣ FEATURE ENGINEERING
# --------------------------------------------------------------------
print("\nPerforming Feature Engineering...")

# Combine train and test for consistent processing
df = pd.concat([train, test], sort=False)

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Create new features
df['family_size'] = df['SibSp'] + df['Parch'] + 1

def transform_family_size(num):
    if num == 1:
        return 'alone'
    elif num <= 4:
        return 'small'
    else:
        return 'large'

df['family_type'] = df['family_size'].apply(transform_family_size)
df['individual_fare'] = df['Fare'] / df['family_size']

# Encode categorical variables
le_dict = {}
for col in ['Sex', 'Embarked', 'family_type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# --------------------------------------------------------------------
# 4️⃣ SPLIT BACK INTO TRAIN & TEST
# --------------------------------------------------------------------
train_clean = df[df['Survived'].notnull()].copy()
test_clean = df[df['Survived'].isnull()].copy()

# Features for model
features = [
    'Pclass', 'Sex', 'Age', 'Fare',
    'family_size', 'individual_fare', 'Embarked'
]

X = train_clean[features]
y = train_clean['Survived'].astype(int)
X_test = test_clean[features]

# Handle missing values
X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X.median(numeric_only=True))

# --------------------------------------------------------------------
# 5️⃣ TRAIN MODEL
# --------------------------------------------------------------------
print("\nTraining Random Forest model...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}")

# --------------------------------------------------------------------
# 6️⃣ GENERATE SUBMISSION
# --------------------------------------------------------------------
print("\nGenerating Kaggle submission file...")

test_clean['Survived'] = model.predict(X_test).astype(int)
submission = test_clean[['PassengerId', 'Survived']]

# Validation checks
if submission.shape[1] != 2:
    raise ValueError("submission.csv must have 2 columns: PassengerId and Survived")
if submission['PassengerId'].duplicated().any():
    raise ValueError("Duplicate PassengerId found!")

submission.to_csv("submission.csv", index=False)
print("\nsubmission.csv created successfully!")
# print("Upload it to: https://www.kaggle.com/competitions/titanic/submit")

# print("\nSample of your submission:")
# print(submission.head(10))
