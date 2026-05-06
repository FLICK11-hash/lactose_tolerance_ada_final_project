import pandas as pd
import numpy as np


def generate_lactose_data(n=1000, seed=42):
    np.random.seed(seed)

    genotype = np.random.choice(
        ["CC", "CT", "TT"],
        size=n,
        p=[0.45, 0.40, 0.15]
    )

    age = np.random.randint(18, 80, size=n)

    dairy_intake = np.random.normal(loc=8, scale=4, size=n)
    dairy_intake = np.clip(dairy_intake, 0, 25)

    family_history = np.random.choice([0, 1], size=n, p=[0.55, 0.45])

    # Risk is highest for CC, lower for CT, lowest for TT
    genotype_risk = {
        "CC": 0.75,
        "CT": 0.40,
        "TT": 0.15
    }

    risk = np.array([genotype_risk[g] for g in genotype])

    # Add small effects from age, dairy intake, and family history
    risk += (age > 50) * 0.05
    risk += (dairy_intake > 10) * 0.05
    risk += family_history * 0.10

    risk = np.clip(risk, 0, 1)

    lactose_intolerant = np.random.binomial(1, risk)

    symptoms_score = (
        lactose_intolerant * np.random.normal(7, 1.5, size=n)
        + (1 - lactose_intolerant) * np.random.normal(3, 1.2, size=n)
    )
    symptoms_score = np.clip(symptoms_score, 0, 10)

    df = pd.DataFrame({
        "rs4988235_genotype": genotype,
        "age": age,
        "dairy_intake_per_week": dairy_intake.round(2),
        "family_history": family_history,
        "symptoms_score": symptoms_score.round(2),
        "lactose_intolerant": lactose_intolerant
    })

    return df


if __name__ == "__main__":
    df = generate_lactose_data()

    df.to_csv("data/simulated_lactose_data.csv", index=False)

    print("Saved simulated dataset to data/simulated_lactose_data.csv")
    print(df.head())
    print()
    print("Class distribution:")
    print(df["lactose_intolerant"].value_counts(normalize=True))