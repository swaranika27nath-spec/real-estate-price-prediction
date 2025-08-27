# real-estate-price-prediction
A machine learning regression project to predict real estate prices using housing data and linear regression algorithms in Python (scikit-learn).
# real_estate_price_prediction.py
# Machine learning regression project to predict real estate prices using scikit-learn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Example: Synthetic data creation for demonstration
# In practice, load your real data with pd.read_csv('housing.csv')
np.random.seed(42)
n_samples = 120
area = np.random.randint(600, 4000, n_samples)
bedrooms = np.random.randint(1, 5, n_samples)
year = np.random.randint(1990, 2021, n_samples)
price = (area * 150) + (bedrooms * 50000) + ((year-1990) * 1000) + np.random.normal(0, 50000, n_samples)

data = pd.DataFrame({
    'area': area,
    'bedrooms': bedrooms,
    'year': year,
    'price': price
})

# Features/labels
X = data[['area', 'bedrooms', 'year']]
y = data['price']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict on the test set
y_pred = reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score (accuracy): {r2:.2f}")

# Example: Predict the price for a new house
sample_house = pd.DataFrame({'area': [2000], 'bedrooms': [3], 'year': [2015]})
predicted_price = reg.predict(sample_house)
print(f"Predicted price for {sample_house.iloc[0].to_dict()}: {predicted_price[0]:,.2f}")
