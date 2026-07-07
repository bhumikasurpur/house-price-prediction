import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


@pytest.fixture(scope="module")
def prepared_data():
    df = pd.read_csv("Housing.csv")
    df = df.dropna(axis=1)

    binary_cols = [
        'mainroad', 'guestroom', 'basement',
        'hotwaterheating', 'airconditioning', 'prefarea'
    ]
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    y = df['price']
    X = df.drop('price', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def trained_model(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def test_no_missing_values_after_cleaning(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0


def test_train_test_split_ratio(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    total = len(X_train) + len(X_test)
    assert abs(len(X_train) / total - 0.8) < 0.02


def test_binary_columns_encoded_correctly(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
        assert set(X_train[col].unique()).issubset({0, 1})


def test_model_predictions_are_positive(trained_model, prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    preds = trained_model.predict(X_test)
    assert all(p > 0 for p in preds), "Predicted house prices should never be negative"


def test_r2_score_above_threshold(trained_model, prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    preds = trained_model.predict(X_test)
    r2 = r2_score(y_test, preds)
    assert r2 > 0.5, f"R2 score {r2} dropped below acceptable threshold"


def test_rmse_within_reasonable_range(trained_model, prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    preds = trained_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    assert rmse < 2_000_000, f"RMSE {rmse} is unexpectedly high"
