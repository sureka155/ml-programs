import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
X = np.array([
    [25, 120],
    [30, 130],
    [35, 140],
    [40, 150],
    [45, 155],
    [55, 190],
    [60, 200],
    [65, 210],
    [70, 220],
    [75, 230]
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
model = SVC(kernel='linear', C=1)
model.fit(X, y)
y_pred = model.predict(X)
 print("Actual Values   :", y)
print("Predicted Values:", y_pred)
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy * 100, "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))
new_patient = np.array([[50, 170]])
prediction = model.predict(new_patient)
if prediction[0] == 0:
    print("\nNew Patient Prediction: No Heart Disease")
else:
    print("\nNew Patient Prediction: Heart Disease")
plt.scatter(X[y == 0, 0], X[y == 0, 1], label="No Heart Disease")
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Heart Disease")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("SVM Heart Disease Classification")
plt.legend()
plt.show()
