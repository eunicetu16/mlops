from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


class ScoringFlow(FlowSpec):

    model_name = Parameter(
        "model_name", default="IrisClassifier", help="Registered model name")

    @step
    def start(self):
        print("Step 1: Ingesting new (unseen) data from a different dataset...")

        # Load Wine dataset (not used in training at all)
        data = load_wine()
        df = pd.DataFrame(data["data"], columns=data["feature_names"])
        df["target"] = data["target"]

        self.holdout = df.sample(n=10, random_state=123).reset_index(drop=True)
        self.X_raw = self.holdout.drop("target", axis=1)
        self.y_true = self.holdout["target"].tolist()

        print(f"Loaded {len(self.X_raw)} rows of new data from Wine dataset.")
        self.next(self.transform)

    @step
    def transform(self):
        print("Step 2: Applying basic feature scaling (standardization)...")

        scaler = StandardScaler()
        self.X_processed = scaler.fit_transform(self.X_raw)

        self.next(self.load_model)

    @step
    def load_model(self):
        print(
            f"Step 3: Loading trained model from MLflow registry: {self.model_name}")

        # Use local MLflow URI
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        model_uri = f"models:/{self.model_name}/latest"
        self.model = mlflow.sklearn.load_model(model_uri)

        self.next(self.predict)

    @step
    def predict(self):
        print("Step 4: Making predictions on new data...")

        try:
            self.predictions = self.model.predict(self.X_processed).tolist()
        except Exception as e:
            self.predictions = []
            print(
                f"Prediction failed due to mismatched feature dimensions: {e}")

        print("\n--- Predictions vs Ground Truth ---")
        for i, (true, pred) in enumerate(zip(self.y_true, self.predictions)):
            print(f"Sample {i}: True = {true}, Predicted = {pred}")

        self.next(self.end)

    @step
    def end(self):
        print("Step 5: Scoring flow completed.")


if __name__ == '__main__':
    ScoringFlow()
