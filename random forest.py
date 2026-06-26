import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

# --------------------------------------------
# Student Dataset (15 Records)
# --------------------------------------------

data = {
    'Study_Hours': [2,3,5,1,4,6,5,2,7,8,3,6,4,5,7],

    'Attendance': [60,70,90,50,80,95,85,65,98,99,75,92,82,88,96],

    'Assignments': [3,4,8,2,6,9,8,4,10,10,5,9,7,8,10],

    'Previous_Marks': [45,50,70,35,65,85,78,48,90,95,60,88,72,80,93],

    'Result': [
        'Fail','Fail','Pass','Fail','Pass',
        'Pass','Pass','Fail','Pass','Pass',
        'Fail','Pass','Pass','Pass','Pass'
    ]
}

df = pd.DataFrame(data)

print("Student Dataset\n")
print(df)

# --------------------------------------------
# Encode Target Variable
# --------------------------------------------

encoder = LabelEncoder()

df['Result'] = encoder.fit_transform(df['Result'])

# --------------------------------------------
# Features and Target
# --------------------------------------------

X = df[['Study_Hours',
        'Attendance',
        'Assignments',
        'Previous_Marks']]

y = df['Result']

# --------------------------------------------
# Train Random Forest Model
# --------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# --------------------------------------------
# Training Accuracy
# --------------------------------------------

accuracy = model.score(X, y)

print("\nTraining Accuracy:", round(accuracy * 100, 2), "%")

# --------------------------------------------
# User Input
# --------------------------------------------

print("\n========== Student Details ==========\n")

study = float(input("Enter Study Hours per Day : "))
attendance = float(input("Enter Attendance (%) : "))
assignments = int(input("Enter Assignments Completed : "))
marks = float(input("Enter Previous Marks : "))

student = pd.DataFrame({
    'Study_Hours':[study],
    'Attendance':[attendance],
    'Assignments':[assignments],
    'Previous_Marks':[marks]
})

prediction = model.predict(student)

result = encoder.inverse_transform(prediction)

print("\n========== Prediction ==========\n")

if result[0] == "Pass":
    print("Student is likely to PASS")
else:
    print("Student is likely to FAIL")

# --------------------------------------------
# Feature Importance Graph
# --------------------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    X.columns,
    model.feature_importances_
)

plt.title("Feature Importance in Random Forest")

plt.xlabel("Features")

plt.ylabel("Importance Score")

plt.grid(axis='y')

plt.show()

# --------------------------------------------
# Confusion Matrix
# --------------------------------------------

y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# --------------------------------------------
# Decision Tree Visualization
# --------------------------------------------

plt.figure(figsize=(20,10))

tree.plot_tree(
    model.estimators_[0],
    feature_names=X.columns,
    class_names=encoder.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("One Decision Tree from the Random Forest")

plt.show()