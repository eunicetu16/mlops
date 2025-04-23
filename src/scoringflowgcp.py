from metaflow import FlowSpec, step, Parameter, kubernetes, conda_base, retry, timeout, catch, resources

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import os


@conda_base(
    python="3.10",
    libraries={
        "pandas": "1.5.3",
        "scikit-learn": "1.2.2",
        "mlflow": "2.2.2",
        "numpy": "1.23.5",
        "imbalanced-learn": "0.10.1",
        "gcsfs": "2023.6.0",
    }
)
class TrainingflowgcpFlow(FlowSpec):
    seed = Parameter("seed", default=42)
    test_size = Parameter("test_size", default=0.2)

    @step
    def start(self):
        df = pd.read_csv("data/cars24data.csv")  # Place file in GCS or volume mount for Kubernetes
        df = df.select_dtypes(include=["float64", "int64"]).dropna()
        self.target_col = "selling_price" if "selling_price" in df.columns else df.columns[-1]
        self.X = df.drop(self.target_col, axis=1)
        self.y = df[self.target_col]
        self.next(self.preprocess)

    @resources(cpu=1, memory=2000)
    @retry(times=2)
    @timeout(seconds=300)
    @step
    def preprocess(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.seed
        )
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train.values
        self.y_test = y_test.values
        joblib.dump(scaler, "scaler.joblib")
        self.next(self.train)

    @kubernetes(cpu=1, memory=2048)
    @retry(times=2)
    @timeout(seconds=300)
    @catch(var="training_error")
    @step
    def train(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.model = model
        self.next(self.evaluate)

    @step
    def evaluate(self):
        preds = self.model.predict(self.X_test)
        self.rmse = float(np.sqrt(mean_squared_error(self.y_test, preds)))
        print(f"Test RMSE: {self.rmse:.2f}")
        self.next(self.register)

    @step
    def register(self):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"))
        mlflow.set_experiment("used_car_price_prediction")
        with mlflow.start_run():
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_metric("rmse", self.rmse)
            mlflow.sklearn.log_model(
                self.model, artifact_path="model", registered_model_name="UsedCarPriceRegressor"
            )

        print("Model registered to MLflow.")
        self.next(self.end)

    @step
    def end(self):
        print("Flow completed.")


if __name__ == "__main__":
    TrainingflowgcpFlow()
