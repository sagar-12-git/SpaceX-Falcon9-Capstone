
import pandas as pd

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("dataset_part_2.csv")

# Target variable
y = df["Class"]

# Use features available before launch.
# Exclude identifiers and direct landing-related fields.
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4, random_state=42
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring="accuracy"
    )

    print(f"\n{name}")
    print("Fold accuracies (%):", (scores * 100).round(2))
    print(f"Mean cross-validation accuracy: {scores.mean() * 100:.2f}%")
    print(f"Standard deviation: {scores.std() * 100:.2f}%")

print("\nFinal independent test set size:", len(X_test))
print("Features used:", X.columns.tolist())
