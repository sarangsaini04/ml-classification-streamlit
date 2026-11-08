import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="ML Classifier Dashboard", page_icon="📊", layout="wide")
st.title("📊 Classification Model Evaluation Dashboard")
st.caption("Upload test data, pick a model, and inspect its performance. Models are pre-trained; no retraining happens here.")

TARGET = "income"   # <- must match train_models.py

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_encoder():
    return joblib.load("model/label_encoder.joblib")

def compute_metrics(pipe, X, y, n_classes):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    avg = "binary" if n_classes == 2 else "weighted"
    if n_classes == 2:
        auc = roc_auc_score(y, proba[:, 1])
    else:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted")
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average=avg, zero_division=0),
        "Recall": recall_score(y, pred, average=avg, zero_division=0),
        "F1": f1_score(y, pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }

le = load_encoder()
n_classes = len(le.classes_)

# ---- Sidebar ----
st.sidebar.header("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV (features + target column)", type=["csv"])
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))

if uploaded is None:
    st.info("👈 Upload `test_data.csv` from the repo to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Preview of uploaded data")
st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
st.dataframe(df.head(10), use_container_width=True)

if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not found in the uploaded file.")
    st.stop()

pipe = load_model(MODEL_FILES[model_name])
required = list(pipe.feature_names_in_)
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()

X = df[required]
try:
    y = le.transform(df[TARGET])
except ValueError:
    st.error("Target column contains labels not seen during training.")
    st.stop()

pred, m = compute_metrics(pipe, X, y, n_classes)

# ---- Metrics ----
st.subheader(f"Evaluation metrics: {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.4f}")

# ---- Confusion matrix + report ----
left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y, pred, target_names=le.classes_.astype(str),
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# ---- All-model comparison on uploaded data ----
st.subheader("Comparison of all models on this test data")
rows = []
for name, path in MODEL_FILES.items():
    _, mm = compute_metrics(load_model(path), X, y, n_classes)
    rows.append({"ML Model Name": name, **mm})
st.dataframe(pd.DataFrame(rows).set_index("ML Model Name").round(4),
             use_container_width=True)
