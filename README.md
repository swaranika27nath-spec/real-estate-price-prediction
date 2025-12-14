# Real Estate Price Prediction

A machine learning project to predict real estate prices using linear regression on synthetic housing data.

## Features

- Generates synthetic dataset with features: area, bedrooms, age
- Trains a Linear Regression model using scikit-learn
- Evaluates model performance with MSE and R² score
- Predicts price for a new house sample

## Installation

1. Ensure Python 3.7+ is installed.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script:
```
python real_estate_price_prediction.py
```

The script will output:
- Mean Squared Error
- R² Score
- Predicted price for a sample house (1500 sq ft, 3 bedrooms, 10 years old)

## Dependencies

- numpy
- pandas
- scikit-learn

## Troubleshooting

- If you encounter import errors, ensure all dependencies are installed.
- The model uses synthetic data; for real predictions, replace with actual dataset.
