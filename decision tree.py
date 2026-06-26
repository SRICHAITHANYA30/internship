import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# ----------------------------------
# Weather Dataset
# ----------------------------------

data = {
    'Outlook': [
        'Sunny','Sunny','Overcast','Rain','Rain',
        'Rain','Overcast','Sunny','Sunny','Rain',
        'Sunny','Overcast','Overcast','Rain',
        'Sunny','Rain','Overcast','Sunny','Rain','Overcast'
    ],

    'Temperature': [
        'Hot','Hot','Hot','Mild','Cool',
        'Cool','Cool','Mild','Cool','Mild',
        'Mild','Mild','Hot','Mild',
        'Cool','Hot','Mild','Hot','Cool','Cool'
    ],

    'Humidity': [
        'High','High','High','High','Normal',
        'Normal','Normal','High','Normal','Normal',
        'Normal','High','Normal','High',
        'High','Normal','High','Normal','High','Normal'
    ],

    'Wind': [
        'Weak','Strong','Weak','Weak','Weak',
        'Strong','Strong','Weak','Weak','Weak',
        'Strong','Strong','Weak','Strong',
        'Weak','Weak','Strong','Strong','Strong','Weak'
    ],

    'Play': [
        'No','No','Yes','Yes','Yes',
        'No','Yes','No','Yes','Yes',
        'Yes','Yes','Yes','No',
        'Yes','Yes','Yes','No','No','Yes'
    ]
}

df = pd.DataFrame(data)

# ----------------------------------
# Label Encoding
# ----------------------------------

le_outlook = LabelEncoder()
le_temp = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

df['Outlook'] = le_outlook.fit_transform(df['Outlook'])
df['Temperature'] = le_temp.fit_transform(df['Temperature'])
df['Humidity'] = le_humidity.fit_transform(df['Humidity'])
df['Wind'] = le_wind.fit_transform(df['Wind'])
df['Play'] = le_play.fit_transform(df['Play'])

# ----------------------------------
# Features and Target
# ----------------------------------

X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = df['Play']

# ----------------------------------
# Train Decision Tree Model
# ----------------------------------

model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

model.fit(X, y)

# ----------------------------------
# User Input
# ----------------------------------

print("\nEnter Weather Details")

outlook = input("Outlook (Sunny/Overcast/Rain): ")
temperature = input("Temperature (Hot/Mild/Cool): ")
humidity = input("Humidity (High/Normal): ")
wind = input("Wind (Weak/Strong): ")

# Encode User Inputs

outlook_encoded = le_outlook.transform([outlook])[0]
temperature_encoded = le_temp.transform([temperature])[0]
humidity_encoded = le_humidity.transform([humidity])[0]
wind_encoded = le_wind.transform([wind])[0]

# ----------------------------------
# Prediction
# ----------------------------------

prediction = model.predict([[
    outlook_encoded,
    temperature_encoded,
    humidity_encoded,
    wind_encoded
]])

result = le_play.inverse_transform(prediction)

print("\nPrediction Result:")
print("Play =", result[0])

# ----------------------------------
# Graph 1 : Decision Tree
# ----------------------------------

plt.figure(figsize=(15, 8))

tree.plot_tree(
    model,
    feature_names=[
        'Outlook',
        'Temperature',
        'Humidity',
        'Wind'
    ],
    class_names=['No', 'Yes'],
    filled=True
)

plt.title("Weather Prediction Decision Tree")
plt.show()

# ----------------------------------
# Graph 2 : Feature Importance
# ----------------------------------

importance = model.feature_importances_

plt.figure(figsize=(8,5))

plt.bar(
    ['Outlook', 'Temperature', 'Humidity', 'Wind'],
    importance
)

plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance")

plt.grid(axis='y')

plt.show()