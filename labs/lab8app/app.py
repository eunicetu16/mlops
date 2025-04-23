from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc

class InputData(BaseModel):
    Manufacturing_year: int
    Engine_capacity: int
    KM_driven: int
    Ownership: int
    Imperfections: int
    Repainted_Parts: int

app = FastAPI()

# Load the MLflow model
model = mlflow.pyfunc.load_model("mlruns/0/034dbcce40094f73b74d46b3c47fa96e/artifacts/cars_model")
@app.get("/")
def root():
    return {"message": "FastAPI ML model is up and running!"}

@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"predicted_price": float(prediction[0])}
