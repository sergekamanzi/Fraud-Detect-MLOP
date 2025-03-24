from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model and scaler
model = tf.keras.models.load_model('fraud_detection_model.keras')
scaler = joblib.load('standard_scaler.pkl')

# Define numeric columns for scaling
numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Define input data schema
class Transaction(BaseModel):
    step: float
    type: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud Detection API. Use /docs for API documentation."}

# Prediction endpoint
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        # Convert input to DataFrame
        input_data = transaction.dict()
        user_input = pd.DataFrame([input_data], columns=['step', 'type', 'amount', 'oldbalanceOrg', 
                                                         'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])
        
        # Scale numeric columns
        user_input[numeric_cols] = scaler.transform(user_input[numeric_cols])
        
        # Make prediction
        prediction = (model.predict(user_input) > 0.5).astype("int32")[0, 0]
        
        # Return result
        result = "Fraud" if prediction == 1 else "No Fraud"
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)