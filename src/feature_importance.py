import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("data/simulated_lactose_data.csv")     # Grabs simulated data

X = df.drop(columns=["lactose_intolerant"])             # Input features, like genotype, age, symptoms, etc.
y = df["lactose_intolerant"]           # Target variable: whether someone is lactose intolerant

categorical_features = ["rs4988235_genotype"]
numeric_features = [
    "age",
    "dairy_intake_per_week",
    "family_history",
    "symptoms_score",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

model = RandomForestClassifier(random_state=42)

pipeline = Pipeline(        # This combines preprocessing and the model into one object.
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
    stratify=y,
)

# Trains the Random Forest model
pipeline.fit(X_train, y_train)

encoded_feature_names = (
    pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

feature_importance = (          # This gets how important each feature was to the Random Forest.
    pipeline.named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({      # This creates the table we are printing here
    "feature": encoded_feature_names,
    "importance": feature_importance,
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False,
)

importance_df.to_csv("data/feature_importance.csv", index=False)     # This saves the feature importance values to a CSV file.

print()
print("Feature importance:")
print(importance_df)

plt.figure(figsize=(10, 6))

plt.barh(           # Creates a horizontal bar chart showing feature importance.
    importance_df["feature"],
    importance_df["importance"],
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()        # puts the most important feature at the top.

plt.tight_layout()
plt.savefig("plots/feature_importance.png", bbox_inches="tight")

print()
print("Saved feature importance table to data/feature_importance.csv")
print("Saved feature importance plot to plots/feature_importance.png")