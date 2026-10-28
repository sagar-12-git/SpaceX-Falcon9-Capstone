
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("dataset_part_2.csv")

y = df["Class"]

excluded = [
    "Class", "Date", "Serial", "LandingPad",
    "Outcome", "Flights", "ReusedCount"
]
X = df.drop(columns=excluded, errors="ignore")

numeric_cols = X.select_dtypes(include=["number"]).columns
categorical_cols = X.select_dtypes(exclude=["number"]).columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train on all rows for the demonstration model
model.fit(X, y)

# Save model and feature names
joblib.dump(
    {"model": model, "features": X.columns.tolist()},
    "spacex_landing_model.joblib"
)

print("Model saved successfully!")
print("File: spacex_landing_model.joblib")
print("Training rows:", len(X))
print("Features:", X.columns.tolist())
