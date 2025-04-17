from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os


class TrainingFlow(FlowSpec):

    seed = Parameter("seed", default=42)
    test_size = Parameter("test_size", default=0.2)
    max_iter = Parameter("max_iter", default=100)

    @step
    def start(self):
        print("Loading Iris dataset")
        data = load_iris()
        df = pd.DataFrame(data["data"], columns=data["feature_names"])
        df["target"] = data["target"]
        self.df = df
        self.next(self.preprocess)

    @step
    def preprocess(self):
        print("Splitting and scaling data")

        X = self.df.drop("target", axis=1)
        y = self.df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Store for future steps
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train.tolist()
        self.y_test = y_test.tolist()

        # Optional: save scaler if needed
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(scaler, "artifacts/scaler.joblib")

        self.next(self.train)

    @step
    def train(self):
        print("Training Logistic Regression")
        model = LogisticRegression(
            max_iter=self.max_iter, random_state=self.seed)
        model.fit(self.X_train, self.y_train)
        self.model = model
        self.next(self.evaluate)

    @step
    def evaluate(self):
        print("Evaluating model")

        y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {self.accuracy:.4f}")
        self.next(self.register)

    @step
    def register(self):
        print("Registering model with MLflow")

        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment("metaflow_training_example")

        with mlflow.start_run():
            mlflow.log_param("max_iter", self.max_iter)
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_metric("accuracy", self.accuracy)
            mlflow.sklearn.log_model(
                self.model,
                artifact_path="model",
                registered_model_name="IrisClassifier"
            )
        print("Model registered successfully.")
        self.next(self.end)

    @step
    def end(self):
        print("Training pipeline completed")


if __name__ == "__main__":
    TrainingFlow()
