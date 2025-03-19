import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

diabetics_dataset = pd.read_csv("/content/diabetes.csv")

cols_to_replace = ["Insulin", "SkinThickness", "BloodPressure"]
for col in cols_to_replace:
   diabetics_dataset[col].replace(0, diabetics_dataset[col].median(), inplace=True)

print(diabetics_dataset["Outcome"].value_counts())

x = diabetics_dataset.drop("Outcome", axis=1)
y = diabetics_dataset["Outcome"]

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

classifier = svm.SVC(kernel="linear", class_weight="balanced")
classifier.fit(x_train_balanced, y_train_balanced)

x_train_prediction = classifier.predict(x_train_balanced)
training_data_accuracy = accuracy_score(x_train_prediction, y_train_balanced)
print("Training Accuracy:", training_data_accuracy)

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Test Accuracy:", test_data_accuracy)

print("Unique Predictions on Test Set:", np.unique(x_test_prediction, return_counts=True))

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train_balanced, y_train_balanced)
rf_test_prediction = rf_classifier.predict(x_test)
rf_test_accuracy = accuracy_score(rf_test_prediction, y_test)
print("Random Forest Test Accuracy:", rf_test_accuracy)

input_data = (12,84,72,31,0,29.7,0.297,46)
input_data_as_numpy_array = np.asarray(input_data).reshape(1,-1)
std_data = scaler.transform(input_data_as_numpy_array)
prediction = classifier.predict(std_data)
print("Prediction:", "Diabetic" if prediction[0] == 1 else "Non-Diabetic")
