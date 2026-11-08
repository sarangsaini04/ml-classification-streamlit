import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
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
    matthews_corrcoef
)

# ---------- CONFIG: change these two ----------
DATA_PATH = "/home/s2026ak05174/workspace/ml-assignment/data.csv"
TARGET = "income" # <- your target column name
# ----------------------------------------------

os.makedirs("model", exist_ok=True)

# The dataset seems to be the Adult Census dataset without headers. 
# We supply standard column names to assign 'income' properly.
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df = pd.read_csv(DATA_PATH, names=column_names, header=None).dropna().reset_index(drop=True)
print("Shape:", df.shape, "| Classes:", df[TARGET].nunique())

X = df.drop(columns=[TARGET])
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
joblib.dump(le, "model/label_encoder.joblib")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Save test data (original labels) for the Streamlit app
test_df = X_test.copy()
test_df[TARGET] = le.inverse_transform(y_test)
test_df.to_csv("test_data.csv", index=False)

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

def make_pipeline(clf):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("prep", pre), ("clf", clf)])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42),
}

n_classes = len(le.classes_)
avg = "binary" if n_classes == 2 else "weighted"

def evaluate(pipe, X_t, y_t):
    pred = pipe.predict(X_t)
    proba = pipe.predict_proba(X_t)
    if n_classes == 2:
        auc = roc_auc_score(y_t, proba[:, 1])
    else:
        auc = roc_auc_score(y_t, proba, multi_class="ovr", average="weighted")
    return {
        "Accuracy": accuracy_score(y_t, pred),
        "AUC": auc,
        "Precision": precision_score(y_t, pred, average=avg, zero_division=0),
        "Recall": recall_score(y_t, pred, average=avg, zero_division=0),
        "F1": f1_score(y_t, pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_t, pred),
    }

rows = []
for name, clf in models.items():
    pipe = make_pipeline(clf)
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, f"model/{name.replace(' ', '_').lower()}.joblib")
    res = evaluate(pipe, X_test, y_test)
    res["ML Model Name"] = name
    rows.append(res)
    print(name, {k: round(v, 4) for k, v in res.items() if k != "ML Model Name"})

results = pd.DataFrame(rows).set_index("ML Model Name").round(4)
print("\n", results)
results.to_csv("model/metrics_comparison.csv")

print("\nMarkdown table for README:\n")
print(results.reset_index().to_markdown(index=False))
