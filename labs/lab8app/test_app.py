import requests

sample = {
    "Manufacturing_year": 2019,
    "Engine_capacity": 1197,
    "KM_driven": 50000,
    "Ownership": 1,
    "Imperfections": 4,
    "Repainted_Parts": 2
}

response = requests.post("http://127.0.0.1:8000/predict", json=sample)
print("Prediction:", response.json())
