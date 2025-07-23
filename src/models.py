import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from src.config import BASE_FEATURES

class BasePriceModel:
    """Base price prediction model - from your Part 1"""
    def __init__(self):
        self.model = None
        self.rmse = None
        
    def train(self, df):
        # Use only non-surge rides
        base_price_data = df[df['surge_multiplier'] == 1.0].copy()
        print(f"Training on {len(base_price_data):,} non-surge rides")
        
        X = base_price_data[BASE_FEATURES]
        y = base_price_data['price']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Base Price Model RMSE: ${self.rmse:.2f}")
        
        return self.model
    
    def predict(self, features):
        return self.model.predict(features)
    
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)