import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("data/simulated_lactose_data.csv")

target = "lactose_intolerant"

feature_sets = {
    "Without symptoms_score": [
        "rs4988235_genotype",
        "age",
        "dairy_intake_per_week",
        "family_history",
    ],
    "With symptoms_score": [
        "rs4988235_genotype",
        "age",
        "dairy_intake_per_week",
        "family_history",
        "symptoms_score",
    ],
}

models = {
    "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=17),
    "Random Forest": RandomForestClassifier(random_state=17),
}

all_results = []

for feature_set_name, features in feature_sets.items():
    X = df[features]
    y = df[target]

    categorical_features = ["rs4988235_genotype"]
    numeric_features = [col for col in features if col != "rs4988235_genotype"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=17,
        stratify=y,
    )

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        all_results.append(
            {
                "feature_set": feature_set_name,
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
        )

results_df = pd.DataFrame(all_results)

print("\nModel comparison:")
print(results_df)

results_df.to_csv("data/model_results.csv", index=False)
print("\nSaved model results to data/model_results.csv")