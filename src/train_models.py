import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("data/simulated_lactose_data.csv")     # Loads data

X = df.drop(columns=["lactose_intolerant"])     # genotype, age, dairy intake, symptoms
y = df["lactose_intolerant"]        # whether someone is lactose intolerant

categorical_features = ["rs4988235_genotype"]      # Identifies categorical and numeric features
numeric_features = ["age", "dairy_intake_per_week", "family_history", "symptoms_score"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),     #Since models cannot directly understand "CC", "CT", and "TT", this converts the genotype text into numbers
        ("num", "passthrough", numeric_features),
    ]
)

models = {                              # The dummy model is the baseline. The others are real ML models discussed about recently in our course!
    "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

X_train, X_test, y_train, y_test = train_test_split(            #80% of the data trains the model. 20% tests how well the model performs on new data. This is a train-test split
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

results = []

for name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    })

results_df = pd.DataFrame(results)

print()
print("Model comparison:")
print(results_df)

results_df.to_csv("data/model_results.csv", index=False)        # This creates a CSV showing which model performed best.
print()
print("Saved model results to data/model_results.csv")