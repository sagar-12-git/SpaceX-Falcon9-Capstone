
import pandas as pd

# Load the SpaceX dataset
df = pd.read_csv("dataset_part_2.csv")

# Display the first 5 rows
print(df.head())

# Display the dataset size
print("\nDataset shape:", df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())
