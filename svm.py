import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("iris.csv")

# Remove Id column
df = df.drop("Id", axis=1)

# Features and Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM Model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# User Input
print("\nEnter Flower Details")

sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width: "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width: "))

sample = [[sl, sw, pl, pw]]

prediction = model.predict(sample)

print("Predicted Species:", encoder.inverse_transform(prediction)[0])