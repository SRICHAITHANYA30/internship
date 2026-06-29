import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("naive.csv")

# Remove Id column
df = df.drop("Id", axis=1)

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Species", axis=1)
y = df["Species"]

# Convert labels into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Naive Bayes Model
# -----------------------------
model = GaussianNB()

model.fit(X_train, y_train)

# -----------------------------
# Accuracy
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# User Input
# -----------------------------
print("\nEnter Flower Details")

sl = float(input("Sepal Length (cm): "))
sw = float(input("Sepal Width (cm): "))
pl = float(input("Petal Length (cm): "))
pw = float(input("Petal Width (cm): "))

sample = pd.DataFrame({
    "SepalLengthCm":[sl],
    "SepalWidthCm":[sw],
    "PetalLengthCm":[pl],
    "PetalWidthCm":[pw]
})

prediction = model.predict(sample)

print("\nPrediction")
print("Flower Species :", encoder.inverse_transform(prediction)[0])