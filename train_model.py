
# CUSTOMER CHURN PREDICTION USING MACHINE LEARNING

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# Load Dataset
data = pd.read_csv("customer_churn.csv")

print("Dataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(data.head())

# Data Cleaning
data.drop("customerID", axis=1, inplace=True)

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

print("\nMissing Values:")
print(data.isnull().sum())

# Separate Input and Output
X = data.drop("Churn", axis=1)
y = data["Churn"].map({"Yes": 1, "No": 0})

# Identify Columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

# Preprocessing
numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Machine Learning Algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

results = {}
trained_models = {}

# Train and Test Models
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy
    trained_models[name] = pipeline

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# Accuracy Comparison
plt.figure(figsize=(9, 5))

plt.bar(results.keys(), results.values())

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

# Select Best Model
best_name = max(results, key=results.get)
best_model = trained_models[best_name]

print("\nBest Model:", best_name)
print("Best Accuracy:", round(results[best_name] * 100, 2), "%")

# Confusion Matrix
predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ROC Curve
plt.figure(figsize=(8, 5))

for name, model in trained_models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test, probabilities
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr, tpr,
        label=f"{name} (AUC = {roc_auc:.2f})"
    )

plt.plot([0, 1], [0, 1], "k--")

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# Save Best Model
joblib.dump(best_model, "churn_model.pkl")

print("\nBest Model Saved Successfully!")
print("ALL TASKS COMPLETED SUCCESSFULLY!")
