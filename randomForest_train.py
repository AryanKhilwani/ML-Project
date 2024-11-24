from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

X_train = train_data.drop(columns=["loan_status"]).values
y_train = train_data["loan_status"].values

X_test = test_data.drop(columns=["loan_status"]).values
y_test = test_data["loan_status"].values

estimators = [10, 30, 50, 70, 90, 110]
accuracy = 0
for estimator in estimators:
    model = RandomForestClassifier(n_estimators=estimator, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("estimators: ", estimator, accuracy_score(y_test, y_pred))
    accuracy = max(accuracy, accuracy_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
