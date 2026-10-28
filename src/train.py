import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

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
        name='churn_model' #just an artifact name... not version name
    )

    print("\nTraining completed!")
    print(f"MLflow Run ID: {run.info.run_id}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-score:  {f1:.4f}")