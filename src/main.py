import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier

# Creates fake dataset bc chromosome 2 seemed to intimidating.
# This is NOT anywhere near my final code for the final submission
# Just brainstorming. For example, if the expected % of people with lactose intolerance is close to the real % in similar populations I know this probably is doing well. Same case for other metrics as well and so forth.

np.random.seed(42)

n = 200

data = pd.DataFrame({
    "SNP_13910": np.random.choice([0, 1], size=n),  # genetic marker
    "age": np.random.randint(18, 80, size=n),
})

# simulate outcome 
data["lactose_intolerant"] = (
    (data["SNP_13910"] == 0) & (np.random.rand(n) > 0.3)
).astype(int)

data.head()

data.describe()
data["lactose_intolerant"].value_counts()

X = data[["SNP_13910", "age"]]
y = data["lactose_intolerant"]

model = DummyClassifier(strategy="most_frequent")
model.fit(X, y)

model.score(X, y)
