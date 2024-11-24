from logisticRegression_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

X_train = train_data.drop(columns=["loan_status"]).values
y_train = train_data["loan_status"].values

X_test = test_data.drop(columns=["loan_status"]).values
y_test = test_data["loan_status"].values

model_LR = LogisticRegression(lr=0.1)  # change learning rate
model_LR.fit(X_train, y_train)

y_pred = model_LR.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
