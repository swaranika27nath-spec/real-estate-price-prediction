# real_estate_price_prediction.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Synthetic dataset: area, bedrooms, age -> price
np.random.seed(42)
n_samples = 500
area = np.random.randint(600, 4000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
age = np.random.randint(0, 30, n_samples)
price = (area * 3000) + (bedrooms * 500000) - (age * 20000) + np.random.randint(-500000, 500000, n_samples)

data = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "age": age,
    "price": price
})

X = data[["area", "bedrooms", "age"]]
y = data["price"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

# 5. Predict a new sample
new_house = np.array([[1500, 3, 10]])  # area, bedrooms, age
predicted_price = model.predict(new_house)[0]
print(f"Predicted price for new house: ₹{predicted_price:,.2f}")