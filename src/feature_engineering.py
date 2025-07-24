import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder

def reverse_engineer_uber_surge(df):

    uber_data = df[df['cab_type'] == 'Uber'].copy()
    
    for service in uber_data['name'].unique():
        service_data = uber_data[uber_data['name'] == service].copy()
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = poly.fit_transform(service_data[['distance']])
        
        huber = HuberRegressor(epsilon=1.35, max_iter=1000)
        huber.fit(X, service_data['price'])
        
        baseline_price = huber.predict(X)
        service_data['implied_surge'] = service_data['price'] / baseline_price
        service_data['implied_surge'] = service_data['implied_surge'].clip(0.5, 3.0)
        
        uber_data.loc[service_data.index, 'implied_surge'] = service_data['implied_surge']
    
    # update main dataset
    df.loc[uber_data.index, 'surge_multiplier'] = uber_data['implied_surge']
    
    return df

def create_final_features(df):

    le_source = LabelEncoder()
    le_dest = LabelEncoder()
    le_service = LabelEncoder()
    
    df['source_encoded'] = le_source.fit_transform(df['source'])
    df['dest_encoded'] = le_dest.fit_transform(df['destination'])
    df['service_encoded'] = le_service.fit_transform(df['name'])
    df['is_uber'] = (df['cab_type'] == 'Uber').astype(int)
    
    # weather severity score
    df['weather_severity'] = (
        df['rain'].fillna(0) * 3 +
        (100 - df['humidity'].fillna(50)) / 100 +
        np.abs(df['temp'].fillna(50) - 65) / 20 +
        df['wind'].fillna(10) / 20
    ).fillna(0)
    
    df['rush_hour_distance'] = df['is_rush_hour'] * df['distance']
    df['weekend_late_night'] = df['is_weekend'] * df['is_late_night']
    df['rain_rush_hour'] = (df['rain'].fillna(0) > 0.1) * df['is_rush_hour']
    
    # later use
    encoders = {
        'source': le_source,
        'destination': le_dest,
        'service': le_service
    }
    
    return df, encoders