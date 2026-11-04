import json
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
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
DATA_FILE = "Dry_Bean_Dataset.xlsx"
TARGET_COL = "Class"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

dataset = pd.read_excel(DATA_FILE, sheet_name="Dry_Beans_Dataset")
X = dataset.drop(columns=[TARGET_COL])
y_text = dataset[TARGET_COL]

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)
joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB()),
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(probability=True, random_state=RANDOM_STATE)),
    ]),
}

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
    else:
        probabilities = model.decision_function(X_test)

    try:
        auc = roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted")
    except ValueError:
        auc = None

    metrics[name] = {
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "AUC": round(auc, 4) if auc is not None else None,
        "Precision": round(precision_score(y_test, predictions, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, predictions, average="weighted", zero_division=0), 4),
        "F1": round(f1_score(y_test, predictions, average="weighted", zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, predictions), 4),
    }
    joblib.dump(model, os.path.join(MODEL_DIR, model_files[name]))

with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

# Save a representative held-out test CSV for repository submission and Streamlit upload.
test_data = X_test.copy()
test_data[TARGET_COL] = encoder.inverse_transform(y_test)
test_data.to_csv("test_data.csv", index=False)

print("Dataset shape:", dataset.shape)
print(json.dumps(metrics, indent=2))
print("Saved test_data.csv with", len(test_data), "rows")
