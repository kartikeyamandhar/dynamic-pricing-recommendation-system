import pandas as pd
import numpy as np

class SurgePricingEngine:
    """Surge pricing logic - from your Part 3"""
    def __init__(self, base_model, surge_thresholds=None):
        self.base_model = base_model
        self.surge_thresholds = surge_thresholds or {
            'low': 1.0,
            'medium': 1.25,
            'high': 1.5,
            'very_high': 2.0,
            'extreme': 2.5
        }
        self.surge_history = []
        
    def calculate_demand_level(self, hour, location, recent_rides, wait_time=None):
        """Calculate current demand level (0-100)"""
        base_demand = 30
        
        # Time-based adjustments
        if hour in [7,8,9,17,18,19]:  # Rush hours
            base_demand += 20
        elif hour in [22,23,0,1,2]:  # Late night
            base_demand += 15
        elif hour in [12,13]:  # Lunch
            base_demand += 10
            
        # Location adjustments
        high_demand_locations = ['Back Bay', 'Financial District', 'North Station']
        if location in high_demand_locations:
            base_demand += 15
            
        # Recent activity
        if recent_rides > 10:
            base_demand += 20
            
        return min(base_demand, 100)
    
    def get_surge_multiplier(self, demand_level, supply_level=50):
        """Determine surge based on demand/supply ratio"""
        ratio = demand_level / supply_level
        
        if ratio < 1.2:
            return self.surge_thresholds['low']
        elif ratio < 1.5:
            return self.surge_thresholds['medium']
        elif ratio < 2.0:
            return self.surge_thresholds['high']
        elif ratio < 2.5:
            return self.surge_thresholds['very_high']
        else:
            return self.surge_thresholds['extreme']