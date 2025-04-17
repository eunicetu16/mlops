from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


class ScoringFlow(FlowSpec):

    model_name = Parameter(
        "model_name", default="IrisClassifier", help="Registered model name")

    @step
    def start(self):
        print("Load Iris test data for scoring")

        # Use the same dataset and features as training
        data = load_iris()
        df = pd.DataFrame(data["data"], columns=data["feature_names"])
        df["target"] = data["target"]

        from sklearn.model_selection import train_test_split
        _, self.df_test = train_test_split(df, test_size=0.2, random_state=42)

        self.X_raw = self.df_test.drop("target", axis=1).copy()
        self.y_true = self.df_test["target"].tolist()

        print(f"Loaded {len(self.X_raw)} rows for testing.")
        self.next(self.transform)

    @step
    def transform(self):
        print("Apply standardization using training distribution")

        # Load full training data to get same scaler statistics
        data = load_iris()
        df = pd.DataFrame(data["data"], columns=data["feature_names"])
        df["target"] = data["target"]
        X_train = df.drop("target", axis=1)

        scaler = StandardScaler()
        scaler.fit(X_train)  # fit only on training set
        self.X_processed = scaler.transform(self.X_raw)

        self.next(self.load_model)

    @step
    def load_model(self):
        print(f"Load model from MLflow registry: {self.model_name}")

        mlflow.set_tracking_uri("file:///tmp/mlruns")
        model_uri = f"models:/{self.model_name}/latest"
        self.model = mlflow.sklearn.load_model(model_uri)

        self.next(self.predict)

    @step
    def predict(self):
        print("Predict on standardized test data")

        self.predictions = self.model.predict(self.X_processed).tolist()

        print("\n--- Predictions vs Ground Truth ---")
        for i, (true, pred) in enumerate(zip(self.y_true, self.predictions)):
            print(f"Sample {i}: True = {true}, Predicted = {pred}")

        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow complete")


if __name__ == '__main__':
    ScoringFlow()
