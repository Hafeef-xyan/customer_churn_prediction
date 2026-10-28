import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
)

df = pd.read_csv('../data/customer_churn_data.csv')

x = df.drop('Churn',axis=1)
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment('customer_churn_prediction')

with mlflow.start_run() as run:

    model.fit(x_train,y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Track parameters
    mlflow.log_params({
        "model_type":"LogisticRegression",
        "C":1.0,
        "max_iter":1000,
        "random_state":42,
        "test_size":0.2
    })

    # Track evaluation metrics
    mlflow.log_metrics({
        "accuracy":accuracy,
        "precision":precision,
        "f1_score":f1
    })

    #save model
    mlflow.sklearn.log_model(
        sk_model=model,
        name='churn_model' #just an artifact name... not version name,
        registered_model_name="CustomerChurnModel" #here model will be registered.. no need to seperate registering.
    )

    #promote if F1 > 0.80
    client = MlflowClient()

    if f1 > 0.80:

        versions = client.search_model_versions(
            f"name='CustomerChurnModel'"
        )

        latest = max(versions, key=lambda v: int(v.version))

        client.set_registered_model_alias(
            "CustomerChurnModel","production",latest.version
        )

        print("Model Promoted to production")
    else:
        print("F1-score below 0.80. Model not promoted.")

    print("\nTraining completed!")
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score:  {f1:.4f}")


'''
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score


DATA_PATH = "data/customer_churn_data.csv"


# Stage 1: Data preparation
def prepare_data():
    df = pd.read_csv(DATA_PATH)

    x = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Encode common Yes/No labels
    if y.dtype == "object":
        y = y.str.strip().str.lower().map(
            {"yes": 1, "no": 0}
        )

    numeric = x.select_dtypes(include="number").columns
    categorical = x.select_dtypes(exclude="number").columns

    preprocessing = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    return preprocessing, X_train, X_test, y_train, y_test


# Stage 2: Training
def train_model(preprocessing, X_train, y_train):
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )),
    ])

    model.fit(X_train, y_train)
    return model


# Stage 3: Evaluation
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
    }

    mlflow.log_metrics(metrics)

    print("\nEvaluation results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("customer_churn_prediction")

    with mlflow.start_run():
        preprocessing, X_train, X_test, y_train, y_test = (
            prepare_data()
        )

        model = train_model(preprocessing, X_train, y_train)

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "C": 1.0,
            "max_iter": 1000,
            "test_size": 0.2,
            "random_state": 42,
        })

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="churn_model",
            registered_model_name="CustomerChurnModel",
        )

        if metrics["f1_score"] > 0.80:
            print("F1 threshold passed.")
        else:
            print("F1 threshold not met.")


if __name__ == "__main__":
    main()
'''