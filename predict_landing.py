
import joblib
import pandas as pd

# Load the saved model
saved = joblib.load("spacex_landing_model.joblib")
model = saved["model"]
features = saved["features"]

# Use one example row from the dataset to demonstrate prediction
df = pd.read_csv("dataset_part_2.csv")
example = df[features].iloc[[0]]

# Predict landing outcome
prediction = model.predict(example)[0]
probabilities = model.predict_proba(example)[0]

print("Example launch features:")
print(example.to_string(index=False))

print("\nPredicted class:", prediction)
print(
    "Predicted outcome:",
    "Successful landing" if prediction == 1
    else "Unsuccessful landing"
)
print(f"Model probability for successful landing: {probabilities[1] * 100:.2f}%")
