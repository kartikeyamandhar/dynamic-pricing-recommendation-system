from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import PPO
import sys,os
from pathlib import Path

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.config import SURGE_LEVELS
from src.surge_engine import SurgePricingEngine

app = FastAPI(title="Uber Dynamic Pricing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models...")

BASE_DIR = Path(__file__).resolve().parent.parent  
MODEL_DIR = BASE_DIR / "models" / "saved_models"

base_model = joblib.load(MODEL_DIR / "base_price_model.pkl")
encoders = joblib.load(MODEL_DIR / "encoders.pkl")
rl_model = PPO.load(str(MODEL_DIR / "uber_pricing_rl_model"))

# base_model = joblib.load("../models/saved_models/base_price_model.pkl")
# encoders = joblib.load("../models/saved_models/encoders.pkl")
# rl_model = PPO.load("../models/saved_models/uber_pricing_rl_model")
surge_engine = SurgePricingEngine(base_model)

class PredictionRequest(BaseModel):
    distance: float
    source: str
    destination: str
    service_type: str
    hour: int = None
    current_demand: float = 50
    current_supply: float = 50

class PredictionResponse(BaseModel):
    base_price: float
    surge_multiplier: float
    final_price: float
    demand_level: float
    supply_level: float
    recommendation: str

@app.get("/")
def root():
    return {"message": "Uber Dynamic Pricing API is running!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    try:
       
        if request.hour is None:
            request.hour = pd.Timestamp.now().hour
        
        state = np.array([
            request.hour,
            pd.Timestamp.now().dayofweek,
            request.current_demand,
            request.current_supply,
            10,  # recent rides
            5,   # avg wait time
            1.0  # competitor surge
        ], dtype=np.float32)
        
        action, _ = rl_model.predict(state)
        rl_surge = SURGE_LEVELS[action]
        
        # Also get rule-based surge for comparison
        demand_level = surge_engine.calculate_demand_level(
            request.hour, request.source, 10
        )
        rule_surge = surge_engine.get_surge_multiplier(
            demand_level, request.current_supply
        )
        
        # For now using   RL surge
        surge_multiplier = rl_surge
        
        # Calculate base price (simplified for demo)
        # In production i will properly encode the categorical variables
        base_price = 2.5 + (request.distance * 1.5)
        
        # Adjust for service type
        service_multipliers = {
            "UberX": 1.0,
            "Lyft": 1.0,
            "UberXL": 1.5,
            "Lyft XL": 1.5,
            "Black": 2.0,
            "Lux Black": 2.0,
            "Black SUV": 2.5,
            "Lux Black XL": 2.5
        }
        base_price *= service_multipliers.get(request.service_type, 1.0)
        
        final_price = base_price * surge_multiplier
        
        # Recommendation
        if surge_multiplier >= 2.0:
            recommendation = "High surge - consider waiting if possible"
        elif surge_multiplier >= 1.5:
            recommendation = "Moderate surge - typical for this time"
        else:
            recommendation = "Good time to ride - low or no surge"
        
        return PredictionResponse(
            base_price=round(base_price, 2),
            surge_multiplier=surge_multiplier,
            final_price=round(final_price, 2),
            demand_level=request.current_demand,
            supply_level=request.current_supply,
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": True
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)