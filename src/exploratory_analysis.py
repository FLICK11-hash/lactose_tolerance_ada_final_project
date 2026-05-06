import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("data/simulated_lactose_data.csv")

print(df.head())
print()
print(df.describe())        # Shows first few rows, averages, mins, maxs, std, etc



# Lactose intolerance distribution
plt.figure(figsize=(6, 4))

df["lactose_intolerant"].value_counts().plot(kind="bar")        # This shows how many people are 0 = not lactose intolerant, 1 = lactose intolerant 

plt.title("Lactose Intolerance Distribution")
plt.xlabel("Lactose Intolerant")
plt.ylabel("Count")

plt.savefig("plots/class_distribution.png")                # Saves as plots/class_distribution.png


# Genotype distribution
plt.figure(figsize=(6, 4))

df["rs4988235_genotype"].value_counts().plot(kind="bar")         # Tells us how many people have which variant of T's and or C's

plt.title("Genotype Distribution")
plt.xlabel("Genotype")
plt.ylabel("Count")
plt.savefig("plots/genotype_distribution.png")      # Saves as plots/genotype_distribution.png


# Symptoms by genotype
plt.figure(figsize=(6, 4))

df.boxplot(column="symptoms_score", by="rs4988235_genotype")        # This compares symptom scores across genotypes.

plt.title("Symptoms Score by Genotype")
plt.suptitle("")

plt.xlabel("Genotype")
plt.ylabel("Symptoms Score")
plt.savefig("plots/symptoms_by_genotype.png")       # Saves as plots/symptoms_by_genotype.png

print()
print("EDA plots saved to plots/")


# Lactose intolerance rate by genotype

# This finds the percentage of lactose-intolerant people for each genotype (T's and C's).
intolerance_by_genotype = df.groupby("rs4988235_genotype")["lactose_intolerant"].mean()


# What percent of CC people are lactose intolerant?
# What percent of CT people are lactose intolerant?
# What percent of TT people are lactose intolerant?
print()
print("Lactose intolerance rate by genotype:")
print(intolerance_by_genotype)

plt.figure(figsize=(6, 4))

intolerance_by_genotype.plot(kind="bar")        # This visualizes the lactose intolerance rate for each genotype.

plt.title("Lactose Intolerance Rate by Genotype")
plt.xlabel("Genotype")
plt.ylabel("Proportion Lactose Intolerant")

plt.savefig("plots/intolerance_rate_by_genotype.png")