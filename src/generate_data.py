import pandas as pd
import numpy as np


def generate_lactose_data(n=1000, seed=45):     # This creates 1000 fake people by default. seed=42 makes the random data repeatable every time you run it so we don't lose this information/people/data.
    np.random.seed(seed)

    genotype = np.random.choice(        
        # Every person gets two copies of the gene variant, one from each parent. We are randomly generating this. 
        # T is associated with lactase persistence
        # C is associated with higher lactose intolerance risk
        ["CC", "CT", "TT"],
        size=n,
        p=[0.45, 0.40, 0.15]        # 45% of people become CC, 40% become CT, 15% become TT
        # These genotype probabilities are simplified simulation assumptions.
        # They are not meant to exactly match one real population.
        # Simplified synthetic assumption, roughly plausible for some European/mixed-European samples, but not representative of all populations - Used a study from Pub Med to determine this statistic (https://pubmed.ncbi.nlm.nih.gov/27577176/)
        # CC is treated as most common here so the dataset has enough lactose-intolerant cases to model.
        # "CC": high risk
        # "CT": low risk
        # "TT": low risk
    )

    age = np.random.randint(18, 80, size=n)     # Gives each person a random age from 18 to 79.

    dairy_intake = np.random.normal(loc=8, scale=4, size=n)     # Creates a realistic spread of dairy intake per week.
    dairy_intake = np.clip(dairy_intake, 0, 25)                 # clip prevents impossible values like negative dairy intake

    family_history = np.random.choice([0, 1], size=n, p=[0.55, 0.45])   # Randomly assigns whether someone has family history. 0 means none, 1 means family history through parent or sibling

    # Risk is highest for CC, lower for CT, lowest for TT
    # For rs4988235, the T allele is treated as protective/dominant for lactase persistence.
    # Therefore CC has the highest intolerance risk, while CT and TT have lower risk.
    genotype_risk = {
        "CC": 0.75,
        "CT": 0.20,
        "TT": 0.10
    }

    risk = np.array([genotype_risk[g] for g in genotype])

    # Add small effects from age and family history
    risk += (age > 50) * 0.05
    # Family history is modeled as a moderate additional risk factor.
    # It represents having a parent or sibling with lactose intolerance.
    # Since rs4988235 genotype is already included, this effect is kept moderate
    # to avoid double-counting inherited genetic risk.
    risk += family_history * 0.15   # This is a synthetic proxy for parentand sibling history, not a real European percentage.
    risk = np.clip(risk, 0, 1)      # Risk must stay between 0 and 1 because it represents probability.

    lactose_intolerant = np.random.binomial(1, risk)        # This decides whether each person is lactose intolerant. 1 = lactose intolerant, 0 = not lactose intolerant

    # Symptoms are only strongly visible when a person actually consumes enough dairy.
    # If someone rarely eats dairy, they may have a low symptom score even if they are lactose intolerant.
    base_symptoms_score = (
        lactose_intolerant * np.random.normal(7, 1.5, size=n)
        + (1 - lactose_intolerant) * np.random.normal(3, 1.2, size=n)
    )

    # Exposure factor: 0 servings/week means symptoms are mostly hidden,
    # 8 or more servings/week means symptoms can fully appear.
    exposure_factor = np.clip(dairy_intake / 8, 0, 1)

    symptoms_score = base_symptoms_score * exposure_factor

    # Higher dairy intake can make symptoms more noticeable, but it does not cause intolerance.
    symptoms_score += (dairy_intake > 10) * 0.75

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

    df.to_csv("data/simulated_lactose_data.csv", index=False)       # This saves the simulated dataset into your data/ folder.

    print("Saved simulated dataset to data/simulated_lactose_data.csv")
    print(df.head())                                                        # This shows the first few rows and the percentage of people who are lactose intolerant.
    print()
    print("Class distribution:")
    print(df["lactose_intolerant"].value_counts(normalize=True))