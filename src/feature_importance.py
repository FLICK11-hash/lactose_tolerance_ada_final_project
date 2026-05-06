import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("data/simulated_lactose_data.csv")

X = df.drop(columns=["lactose_intolerant"])
y = df["lactose_intolerant"]

categorical_features = ["rs4988235_genotype"]
numeric_features = [
    "age",
    "dairy_intake_per_week",
    "family_history",
    "symptoms_score"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

model = RandomForestClassifier(random_state=42)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

encoded_feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

feature_importance = (
    pipeline.named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "feature": encoded_feature_names,
    "importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print()
print("Feature importance:")
print(importance_df)

plt.figure(figsize=(8, 5))

plt.barh(
    importance_df["feature"],
    importance_df["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.savefig("plots/feature_importance.png")

print()
print("Saved feature importance plot to plots/feature_importance.png")