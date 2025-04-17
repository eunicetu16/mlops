from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib


class ScoringFlow(FlowSpec):

    model_name = Parameter(
        "model_name", default="UsedCarPriceRegressor", help="Registered model name")

    @step
    def start(self):
        # Load and preprocess data
        df = pd.read_csv(
            "/Users/eunicetu/Downloads/MSDS/MSDS603-mlops/data/cars24data.csv")
        df = df.select_dtypes(include=["float64", "int64"]).dropna()

        self.target_col = "selling_price" if "selling_price" in df.columns else df.columns[-1]
        self.holdout = df.sample(n=10, random_state=999).reset_index(drop=True)

        self.X_raw = self.holdout.drop(self.target_col, axis=1)
        self.y_true = self.holdout[self.target_col].tolist()

        print(f"Sample size: {len(self.X_raw)} rows")
        self.next(self.transform)

    @step
    def transform(self):
        scaler = joblib.load("artifacts/scaler.joblib")
        self.X_processed = scaler.transform(self.X_raw)

        self.next(self.load_model)

    @step
    def load_model(self):

        mlflow.set_tracking_uri("file:///tmp/mlruns")
        model_uri = f"models:/{self.model_name}/latest"
        self.model = mlflow.sklearn.load_model(model_uri)

        self.next(self.predict)

    @step
    def predict(self):
        print("Making predictions")

        self.predictions = self.model.predict(self.X_processed).tolist()

        for i, (pred, true) in enumerate(zip(self.predictions, self.y_true)):
            print(f"Row {i}: Predicted = {pred:.2f}, Actual = {true:.2f}")

        from sklearn.metrics import mean_squared_error
        import numpy as np

        rmse = np.sqrt(mean_squared_error(self.y_true, self.predictions))
        print(f"\nRMSE: {rmse:.2f}")
        self.next(self.end)

    @step
    def end(self):
        print("Scoring completed.")


if __name__ == "__main__":
    ScoringFlow()
