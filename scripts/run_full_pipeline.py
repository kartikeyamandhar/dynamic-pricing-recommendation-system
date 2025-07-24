#!/usr/bin/env python

import sys,os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pandas as pd
import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data_preprocessing import load_and_clean_data, create_temporal_features, merge_weather_data
from src.feature_engineering import reverse_engineer_uber_surge, create_final_features
from src.models import BasePriceModel
from src.rl_environment import UberPricingEnv

def main():

    print("Step 1: Loading data...")
    df_rides, df_weather = load_and_clean_data()
    
    print("\nStep 2: Creating temporal features...")
    df_rides = create_temporal_features(df_rides)
    
    print("\nStep 3: Merging with weather data...")
    df_merged = merge_weather_data(df_rides, df_weather)
    
    print("\nStep 4: Reverse-engineering Uber surge...")
    df_merged = reverse_engineer_uber_surge(df_merged)
    
    print("\nStep 5: Creating final features...")
    df_final, encoders = create_final_features(df_merged)
    
    df_final.to_csv('data/processed/uber_lyft_processed.csv', index=False)
    joblib.dump(encoders, 'models/saved_models/encoders.pkl')
    print("Saved processed data and encoders!")
    
    print("\nStep 6: Training base price model...")
    base_model = BasePriceModel()
    base_model.train(df_final)
    base_model.save('models/saved_models/base_price_model.pkl')
    
    print("\nStep 7: Training RL agent...")
    env = UberPricingEnv(df_final, base_model.model)
    env = DummyVecEnv([lambda: env])
    
    rl_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1
    )
    
    rl_model.learn(total_timesteps=300000)
    
    rl_model.save("models/saved_models/uber_pricing_rl_model")
    print("\nmodels trained and saved")
    
if __name__ == "__main__":
    main()