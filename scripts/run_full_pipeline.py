#!/usr/bin/env python
"""
Complete training pipeline - runs all your notebook code in order
"""
import sys
# sys.path.append('.')

import pandas as pd
import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import our modules
from src.data_processing import load_and_clean_data, create_temporal_features, merge_weather_data
from src.feature_engineering import reverse_engineer_uber_surge, create_final_features
from src.models import BasePriceModel
from src.rl_environment import UberPricingEnv

def main():
    print("=== UBER DYNAMIC PRICING - FULL PIPELINE ===\n")
    
    # Step 1: Load and process data
    print("Step 1: Loading data...")
    df_rides, df_weather = load_and_clean_data()
    
    # Step 2: Create temporal features
    print("\nStep 2: Creating temporal features...")
    df_rides = create_temporal_features(df_rides)
    
    # Step 3: Merge with weather
    print("\nStep 3: Merging with weather data...")
    df_merged = merge_weather_data(df_rides, df_weather)
    
    # Step 4: Reverse engineer Uber surge
    print("\nStep 4: Reverse-engineering Uber surge...")
    df_merged = reverse_engineer_uber_surge(df_merged)
    
    # Step 5: Create final features
    print("\nStep 5: Creating final features...")
    df_final, encoders = create_final_features(df_merged)
    
    # Save processed data
    df_final.to_csv('data/processed/uber_lyft_processed.csv', index=False)
    joblib.dump(encoders, 'models/saved_models/encoders.pkl')
    print("Saved processed data and encoders!")
    
    # Step 6: Train base price model
    print("\nStep 6: Training base price model...")
    base_model = BasePriceModel()
    base_model.train(df_final)
    base_model.save('models/saved_models/base_price_model.pkl')
    
    # Step 7: Train RL agent
    print("\nStep 7: Training RL agent...")
    env = UberPricingEnv(df_final, base_model.model)
    env = DummyVecEnv([lambda: env])
    
    # Initialize PPO
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
    
    print("Training for 100,000 timesteps...")
    rl_model.learn(total_timesteps=100000)
    
    # Save RL model
    rl_model.save("models/saved_models/uber_pricing_rl_model")
    print("\nAll models trained and saved!")
    
    print("\n=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()