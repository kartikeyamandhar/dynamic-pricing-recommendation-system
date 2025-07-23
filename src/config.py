# Configuration constants from your notebook
RUSH_HOURS = [7, 8, 9, 17, 18, 19]
LATE_NIGHT_HOURS = [22, 23, 0, 1, 2, 3]
LUNCH_HOURS = [12, 13]
WEEKEND_DAYS = [5, 6]

SURGE_LEVELS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

BASE_FEATURES = [
    'distance', 
    'service_encoded',
    'is_uber',
    'hour',
    'day_of_week'
]

FEATURE_COLUMNS = [
    'distance', 'surge_multiplier',
    'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'is_late_night', 'is_lunch_hour',
    'source_encoded', 'dest_encoded',
    'service_encoded', 'is_uber',
    'temp', 'humidity', 'wind', 'rain', 'pressure', 'weather_severity',
    'rush_hour_distance', 'weekend_late_night', 'rain_rush_hour'
]