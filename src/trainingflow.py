from metaflow import FlowSpec, step, Parameter
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class TrainingFlow(FlowSpec):

    seed = Parameter("seed", default=42)
    test_size = Parameter("test_size", default=0.2)
    max_iter = Parameter("max_iter", default=100)

    @step
    def start(self):
        from sklearn.datasets import load_iris
        print("Starting training pipeline...")

        data = load_iris()
        self.df = pd.DataFrame(data["data"], columns=data["feature_names"])
        self.df["target"] = data["target"]

        self.next(self.preprocess)

    @step
    def preprocess(self):
        print("Preprocessing data...")
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed
        )
        self.next(self.train)

    @step
    def train(self):
        print("Training model...")
        self.model = LogisticRegression(
            max_iter=self.max_iter, random_state=self.seed)
        self.model.fit(self.X_train, self.y_train)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        print("Evaluating model...")
        y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model accuracy: {self.accuracy:.4f}")
        self.next(self.register)

    @step
    def register(self):
        import mlflow
        import mlflow.sklearn

        print("Registering model to MLflow...")

        # Use a local MLflow tracking URI
        # stores runs locally in filesystem
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

        print("Model registered in MLflow!")
        self.next(self.end)

    @step
    def end(self):
        print("Training pipeline complete.")


if __name__ == '__main__':
    TrainingFlow()
