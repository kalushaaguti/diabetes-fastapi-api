# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 23:50:18 2025

@author: kalus
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load model
model = pickle.load(open("trained_model.sav", "rb"))

# Create FastAPI instance
app = FastAPI(
    title="Diabetes Prediction API",
    description="An API that predicts whether a person is diabetic based on medical data.",
    version="1.0"
)

# Define the request body schema
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API!"}

@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    input_data = np.array([
        data.Pregnancies, data.Glucose, data.BloodPressure,
        data.SkinThickness, data.Insulin, data.BMI,
        data.DiabetesPedigreeFunction, data.Age
    ]).reshape(1, -1)

    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    return {"prediction": result}
