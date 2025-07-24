import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(rides_path='data/raw/cab_rides.csv', 
                        weather_path='data/raw/weather.csv'):

    df_rides = pd.read_csv(rides_path)
    df_weather = pd.read_csv(weather_path)
    
    print(f"Rides data: {df_rides.shape}")
    print(f"Weather data: {df_weather.shape}")
    
    df_clean = df_rides[df_rides['price'].notna()].copy()
    print(f"Removed {len(df_rides) - len(df_clean)} rows with missing prices")
    
    return df_clean, df_weather

def create_temporal_features(df):

    df['datetime'] = pd.to_datetime(df['time_stamp'], unit='ms')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['date'] = df['datetime'].dt.date
    
    df['is_rush_hour'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)
    df['is_late_night'] = df['hour'].isin([22,23,0,1,2,3]).astype(int)
    df['is_lunch_hour'] = df['hour'].isin([12,13]).astype(int)
    
    df['distance_category'] = pd.cut(df['distance'], 
                                     bins=[0, 0.5, 1, 2, 3, 10], 
                                     labels=['very_short', 'short', 'medium', 'long', 'very_long'])
    return df

def merge_weather_data(df_rides, df_weather):


    df_weather['datetime'] = pd.to_datetime(df_weather['time_stamp'], unit='s')
    df_weather['hour_timestamp'] = df_weather['datetime'].dt.floor('H')
    
    df_rides['hour_timestamp'] = df_rides['datetime'].dt.floor('H')
    
    df_merged = pd.merge(
        df_rides,
        df_weather,
        left_on=['source', 'hour_timestamp'],
        right_on=['location', 'hour_timestamp'],
        how='left',
        suffixes=('', '_weather')
    )
    
    print(f"Merged dataset shape: {df_merged.shape}")
    print(f"Weather match rate: {df_merged['temp'].notna().mean()*100:.1f}%")
    
    return df_merged