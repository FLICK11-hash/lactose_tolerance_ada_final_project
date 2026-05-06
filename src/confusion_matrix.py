import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

# Quick summary of this py file :
# This script evaluates the Logistic Regression model visually by showing how many 
# test examples it classified correctly and incorrectly.


df = pd.read_csv("data/simulated_lactose_data.csv")     # Loads the data

X = df.drop(columns=["lactose_intolerant"])
y = df["lactose_intolerant"]

categorical_features = ["rs4988235_genotype"]
numeric_features = ["age", "dairy_intake_per_week", "family_history", "symptoms_score"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Trains the logistic regression model
model.fit(X_train, y_train)

# Creates a confusion matrix for the test set
ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    display_labels=["Not Intolerant", "Intolerant"],
)

plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", bbox_inches="tight")

print("Saved confusion matrix to plots/confusion_matrix.png")