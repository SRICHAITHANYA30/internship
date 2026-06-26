import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------------
# Load Dataset
# ----------------------------------

df = pd.read_csv("spam.csv")

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['Label', 'Message']

# Convert labels to numbers
# ham = 0, spam = 1

df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})

# ----------------------------------
# Features and Target
# ----------------------------------

X = df['Message']
y = df['Label']

# ----------------------------------
# Convert text into numbers
# ----------------------------------

vectorizer = TfidfVectorizer(stop_words='english')

X_vector = vectorizer.fit_transform(X)

# ----------------------------------
# Train Logistic Regression Model
# ----------------------------------

model = LogisticRegression()

model.fit(X_vector, y)

# ----------------------------------
# User Input
# ----------------------------------

print("\n========== Spam Mail Detection ==========\n")

user_mail = input("Enter your email message:\n")

# Convert user input to vector
user_vector = vectorizer.transform([user_mail])

# ----------------------------------
# Prediction
# ----------------------------------

prediction = model.predict(user_vector)

print("\n========== Prediction ==========\n")

if prediction[0] == 1:
    print("Result : SPAM MAIL")
else:
    print("Result : NOT SPAM MAIL")