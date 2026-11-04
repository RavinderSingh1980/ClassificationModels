import os
import json
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

TARGET_COL = "Class"
MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Dry Bean Classification", page_icon="Beans", layout="wide")
st.title("Dry Bean Classification System")
st.write("Upload test data to evaluate the five saved machine-learning models. No models are retrained from the uploaded file.")

@st.cache_resource
def load_encoder():
    return joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))

@st.cache_data
def load_training_metrics():
    metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return None

try:
    encoder = load_encoder()
except FileNotFoundError:
    st.error("Missing model/label_encoder.joblib. Run train_models.py before starting the app.")
    st.stop()

metrics = load_training_metrics()
if metrics:
    st.subheader("Training/test performance comparison")
    st.dataframe(pd.DataFrame(metrics).T.rename_axis("Model").reset_index(), use_container_width=True)

uploaded_file = st.file_uploader("Upload test CSV (include all input features and the Class column)", type=["csv"])
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    if TARGET_COL not in test_data.columns:
        st.error(f"The uploaded CSV must contain the target column '{TARGET_COL}'.")
        st.stop()

    y_true_text = test_data[TARGET_COL]
    X_test = test_data.drop(columns=[TARGET_COL])

    missing_classes = sorted(set(y_true_text.astype(str)) - set(map(str, encoder.classes_)))
    if missing_classes:
        st.error("Unknown class labels found: " + ", ".join(missing_classes))
        st.stop()
    y_true = encoder.transform(y_true_text)

    available_models = {
        name: filename for name, filename in MODEL_FILES.items()
        if os.path.exists(os.path.join(MODEL_DIR, filename))
    }
    if not available_models:
        st.error("No trained model files found. Run train_models.py first.")
        st.stop()

    selected_model = st.selectbox("Select a model", list(available_models))
    model = load_model(available_models[selected_model])

    try:
        y_pred = model.predict(X_test)
    except ValueError as error:
        st.error(f"The uploaded CSV does not match the trained feature format: {error}")
        st.stop()

    scores = None
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)

    multiclass_auc = None
    if scores is not None:
        try:
            if len(encoder.classes_) == 2:
                multiclass_auc = roc_auc_score(y_true, scores[:, 1])
            else:
                multiclass_auc = roc_auc_score(y_true, scores, multi_class="ovr", average="weighted")
        except ValueError:
            pass

    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": multiclass_auc if multiclass_auc is not None else "Not available",
        "Precision (weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    st.subheader(f"Evaluation results: {selected_model}")
    metric_columns = st.columns(3)
    for index, (name, value) in enumerate(results.items()):
        if isinstance(value, float):
            metric_columns[index % 3].metric(name, f"{value:.4f}")
        else:
            metric_columns[index % 3].metric(name, str(value))

    labels = list(range(len(encoder.classes_)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)

    st.subheader("Confusion Matrix")
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=encoder.classes_.astype(str),
        zero_division=0,
    )
    st.code(report)
