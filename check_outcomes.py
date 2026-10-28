
import pandas as pd

df = pd.read_csv("dataset_part_2.csv")

print("Landing outcome counts:")
print(df["Class"].value_counts().sort_index())

print("\nLanding success percentage:")
print(round(df["Class"].mean() * 100, 2), "%")

print("\nOutcome descriptions:")
print("Class 1 = Successful landing")
print("Class 0 = Unsuccessful landing")
