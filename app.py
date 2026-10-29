# train_models.py (top section)
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

RANDOM_STATE = 42
DATA_FILE = "Dry_Bean_Dataset.xlsx"
TARGET_COL = "Class"
os.makedirs("model", exist_ok=True)

# 1. Load data from the Excel file
xls = pd.ExcelFile(DATA_FILE)
print("Sheets found:", xls.sheet_names)       # confirm the sheet name
df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])   # or sheet_name="<exact name>"

print(df.shape)
print(df.isnull().sum().sum(), "missing values")
print(df.duplicated().sum(), "duplicate rows")
print(df[TARGET_COL].value_counts())

# 2. Split features / target
X = df.drop(columns=[TARGET_COL])
y_raw = df[TARGET_COL]

# 3. Encode target (keep the encoder for the app)
le = LabelEncoder()
y = pd.Series(le.fit_transform(y_raw), name=TARGET_COL)
joblib.dump(le, "model/label_encoder.joblib")
