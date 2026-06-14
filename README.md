# House Price Prediction (Linear Regression)

A regression model that predicts house prices based on features like 
area, number of bedrooms, bathrooms, parking, and amenities (AC, 
guestroom, furnishing status, etc.).

## Dataset
Housing.csv — 545 records, 13 features (Kaggle housing price dataset).

## Approach
- Cleaned data and handled categorical features via one-hot encoding 
  (furnishingstatus) and binary mapping (mainroad, guestroom, etc.)
- Split data into train/test sets (80/20)
- Trained a Linear Regression model using scikit-learn

## Results
- RMSE: ~1,324,507
- R²: ~0.65

## Files
- `house_price_prediction.ipynb` — full notebook with code and outputs
- `predictions.csv` — actual vs predicted prices on test set
