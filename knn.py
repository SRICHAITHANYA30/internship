import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train KNN Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nEnter Patient Details")

age = int(input("Age: "))
sex = int(input("Sex (1=Male,0=Female): "))
cp = int(input("Chest Pain Type: "))
trestbps = int(input("Resting Blood Pressure: "))
chol = int(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar (1/0): "))
restecg = int(input("Rest ECG: "))
thalach = int(input("Max Heart Rate: "))
exang = int(input("Exercise Induced Angina (1/0): "))
oldpeak = float(input("Oldpeak: "))
slope = int(input("Slope: "))
ca = int(input("CA: "))
thal = int(input("Thal: "))

sample = [[
    age, sex, cp, trestbps, chol,
    fbs, restecg, thalach,
    exang, oldpeak, slope,
    ca, thal
]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease")