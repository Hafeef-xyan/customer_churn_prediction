from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def test_logistic_regression_model():
    # Small sample dataset
    X_train = [[10], [20], [30], [80], [90], [100]]
    y_train = [0, 0, 0, 1, 1, 1]

    # Same model configuration as train.py
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict([[15], [95]])

    # Verify predictions are valid
    assert list(predictions) == [0, 1]

    # Verify model performance on the sample training data
    train_predictions = model.predict(X_train)
    accuracy = accuracy_score(y_train, train_predictions)

    assert accuracy >= 0.80
