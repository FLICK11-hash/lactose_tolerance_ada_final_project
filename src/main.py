import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier

np.random.seed(42)

n = 200

data = pd.DataFrame({
    "SNP_13910": np.random.choice([0, 1], size=n),  # genetic marker
    "age": np.random.randint(18, 80, size=n),
})

# simulate outcome (very rough biological logic)
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
