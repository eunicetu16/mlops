from metaflow import FlowSpec, step, Parameter
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


class TrainingFlow(FlowSpec):

    seed = Parameter("seed", default=42)
    test_size = Parameter("test_size", default=0.2)

    @step
    def start(self):
        df = pd.read_csv(
            "/Users/eunicetu/Downloads/MSDS/MSDS603-mlops/data/cars24data.csv")
        # basic numeric feature selection
        df = df.select_dtypes(include=["float64", "int64"])
        df = df.dropna()

        self.target_col = "selling_price" if "selling_price" in df.columns else df.columns[-1]

        self.X = df.drop(self.target_col, axis=1)
        self.y = df[self.target_col]
        print(f"Data shape: {self.X.shape}")
        self.next(self.preprocess)

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

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(scaler, "artifacts/scaler.joblib")

        self.next(self.train)

    @step
    def train(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.model = model
        self.next(self.evaluate)

    @step
    def evaluate(self):
        preds = self.model.predict(self.X_test)
        self.rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        print(f"Test RMSE: {self.rmse:.2f}")
        self.next(self.register)

    @step
    def register(self):
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("used_car_price_prediction")

        with mlflow.start_run():
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_metric("rmse", self.rmse)
            mlflow.sklearn.log_model(
                self.model,
                artifact_path="model",
                registered_model_name="UsedCarPriceRegressor"
            )
        print("Model logged.")
        self.next(self.end)

    @step
    def end(self):
        print("Training complete.")


if __name__ == '__main__':
    TrainingFlow()
