import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------
# Sample House Dataset (15 Records)
# -----------------------------------

data = {
    'Area': [1000, 1200, 1400, 1600, 1800,
             2000, 2200, 2400, 2600, 2800,
             3000, 3200, 3400, 3600, 3800],

    'Bedrooms': [2, 2, 3, 3, 3,
                 4, 4, 4, 5, 5,
                 5, 6, 6, 6, 7],

    'Bathrooms': [1, 2, 2, 2, 3,
                  3, 3, 4, 4, 4,
                  5, 5, 5, 6, 6],

    'Floors': [1, 1, 1, 2, 2,
               2, 2, 3, 3, 3,
               3, 4, 4, 4, 4],

    'Price': [3000000, 3600000, 4200000, 5000000, 5800000,
              6500000, 7200000, 8000000, 9000000, 9800000,
              10500000, 11500000, 12300000, 13200000, 14000000]
}

df = pd.DataFrame(data)

# -----------------------------------
# Features and Target
# -----------------------------------

X = df[['Area', 'Bedrooms', 'Bathrooms', 'Floors']]
y = df['Price']

# -----------------------------------
# Train Linear Regression Model
# -----------------------------------

model = LinearRegression()
model.fit(X, y)

# -----------------------------------
# Predictions
# -----------------------------------

y_pred = model.predict(X)

# -----------------------------------
# Evaluation
# -----------------------------------

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Model Performance")
print("------------------")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# -----------------------------------
# User Prediction
# -----------------------------------

print("\nEnter House Details")

area = float(input("Area (sq.ft): "))
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))
floors = int(input("Floors: "))

new_house = pd.DataFrame({
    'Area': [area],
    'Bedrooms': [bedrooms],
    'Bathrooms': [bathrooms],
    'Floors': [floors]
})

predicted_price = model.predict(new_house)

print("\nEstimated House Price:")
print(f"₹ {predicted_price[0]:,.2f}")

# -----------------------------------
# Graph: Actual vs Predicted
# -----------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y, y_pred)

# Perfect Prediction Line
plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linewidth=2
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)

plt.show()