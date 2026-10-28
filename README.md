
# SpaceX Falcon 9 First-Stage Landing Prediction

## 1. Project Objective
Build machine-learning models to predict whether a
SpaceX Falcon 9 first-stage landing will be successful.

## 2. Dataset
- Total records: 90
- Total original columns: 18
- Target column: Class
- Class 1: Successful landing
- Class 0: Unsuccessful landing

## 3. Data Exploration
- Missing values were found in LandingPad.
- No duplicate rows were detected.
- The dataset success rate was 66.67%.

## 4. Machine-Learning Models
- Logistic Regression
- Decision Tree Classifier

## 5. Evaluation
Five-fold cross-validation on the training split produced:
- Logistic Regression: 84.76% mean accuracy
- Decision Tree: 80.38% mean accuracy

## 6. Conclusion
Logistic Regression achieved the higher mean
cross-validation accuracy in this experiment.
The dataset is small, so further validation is
needed before using the model for real predictions.

## 7. Technologies Used
- Python 3.13
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Joblib
