import gym
from gym import spaces
import numpy as np
import pandas as pd
from collections import deque

class UberPricingEnv(gym.Env):
    """Your exact RL environment from the notebook"""
    def __init__(self, historical_data, base_price_model, episode_length=100):
        super(UberPricingEnv, self).__init__()
        
        self.data = historical_data
        self.base_model = base_price_model
        self.episode_length = episode_length
        
        self.action_space = spaces.Discrete(7)
        self.surge_levels = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 1.0]),
            high=np.array([23, 6, 100, 100, 50, 30, 3.0]),
            dtype=np.float32
        )
        
        self.revenue_history = deque(maxlen=1000)
        self.acceptance_history = deque(maxlen=1000)
        self.surge_history = deque(maxlen=1000)
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.current_index = np.random.randint(0, len(self.data) - self.episode_length)
        self.episode_revenue = 0
        self.episode_rides = 0
        self.episode_accepted = 0
        
        self.current_demand = np.random.randint(20, 80)
        self.current_supply = np.random.randint(30, 70)
        
        return self._get_state()
    
    def _get_state(self):
        current_data = self.data.iloc[self.current_index]
        
        recent_window = self.data.iloc[max(0, self.current_index-20):self.current_index]
        recent_rides = len(recent_window)
        avg_wait = np.random.uniform(2, 15)
        competitor_surge = np.random.choice([1.0, 1.25, 1.5], p=[0.7, 0.2, 0.1])
        
        state = np.array([
            current_data['hour'],
            current_data['day_of_week'],
            self.current_demand,
            self.current_supply,
            recent_rides,
            avg_wait,
            competitor_surge
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        surge_multiplier = self.surge_levels[action]
        current_ride = self.data.iloc[self.current_index]
        
        base_features = pd.DataFrame({
            'distance': [current_ride['distance']],
            'service_encoded': [current_ride['service_encoded']],
            'is_uber': [current_ride['is_uber']],
            'hour': [current_ride['hour']],
            'day_of_week': [current_ride['day_of_week']]
        })
        base_price = self.base_model.predict(base_features)[0]
        
        final_price = base_price * surge_multiplier
        
        price_sensitivity = 0.3
        acceptance_prob = 1.0 / (1 + price_sensitivity * (surge_multiplier - 1)**2)
        
        if current_ride.get('rain', 0) > 0.1:
            acceptance_prob *= 1.2
        if current_ride['hour'] in [22, 23, 0, 1, 2]:
            acceptance_prob *= 1.1
            
        acceptance_prob = np.clip(acceptance_prob, 0.1, 0.95)
        ride_accepted = np.random.random() < acceptance_prob
        
        reward = self._calculate_reward(
            final_price, surge_multiplier, ride_accepted, 
            acceptance_prob, self.current_demand, self.current_supply
        )

        if ride_accepted:
            self.episode_revenue += final_price
            self.episode_accepted += 1
        self.episode_rides += 1
        
        self._update_market_conditions(surge_multiplier, ride_accepted)
        
        self.current_step += 1
        self.current_index += 1
        
        done = self.current_step >= self.episode_length
        
        self.revenue_history.append(final_price if ride_accepted else 0)
        self.acceptance_history.append(ride_accepted)
        self.surge_history.append(surge_multiplier)
        
        next_state = self._get_state()
        
        info = {
            'revenue': self.episode_revenue,
            'acceptance_rate': self.episode_accepted / max(1, self.episode_rides),
            'avg_surge': np.mean(list(self.surge_history)),
            'demand': self.current_demand,
            'supply': self.current_supply
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, price, surge, accepted, accept_prob, demand, supply):
        reward = 0
        
        if accepted:
            reward += price * 0.5
            if demand > supply and surge > 1.0:
                reward += 5
            elif demand < supply and surge == 1.0:
                reward += 3
        else:
            reward -= 2
            if demand < supply and surge > 1.5:
                reward -= 5
        
        utilization = min(demand / max(supply, 1), 1.0)
        reward += utilization * 2
        
        if surge > 2.0:
            reward -= (surge - 2.0) * 10
        if surge >= 2.5:
            reward -= 10
            
        return reward
    
    def _update_market_conditions(self, surge, accepted):
        if not accepted:
            self.current_demand *= 0.98
        else:
            self.current_demand *= 1.01
            
        if surge > 1.5:
            self.current_supply += np.random.randint(1, 5)
        elif surge == 1.0:
            self.current_supply -= np.random.randint(0, 3)
            
        self.current_demand += np.random.randint(-5, 5)
        self.current_supply += np.random.randint(-3, 3)
        
        self.current_demand = np.clip(self.current_demand, 10, 100)
        self.current_supply = np.clip(self.current_supply, 10, 100)