# Dynamic Pricing with Reinforcement Learning

An intelligent pricing system that uses Reinforcement Learning to optimize surge pricing strategies, balancing revenue, customer satisfaction, and driver utilization.

<img width="548" height="950" alt="Image" src="https://github.com/user-attachments/assets/4e58dbba-47b3-4d27-95f1-b25a47e25381" />

## 🌟 Why This Project is Different

Instead of predicting historical prices, we use **Reinforcement Learning** to learn optimal pricing strategies that balance multiple objectives:

### 🔄 Real Market Dynamics
Our RL agent learns to respond to supply and demand changes, just like a real marketplace

### 🧠 Beyond Simple Regression
We reverse-engineer Uber's surge multipliers and train an AI agent that learns **when** and **how much** to surge

### ⚖️ Multi-Objective Optimization
The system balances revenue, customer satisfaction, and driver utilization - not just accuracy

---

## 🎯 The Theory Behind Our Approach

### Traditional Approach ❌
```
Historical Data → Features → ML Model → Predict Price
```
**Problems:** Static, backward-looking, ignores market dynamics

### Our Approach (Dynamic Pricing Optimization)
```
Market State → RL Agent → Surge Decision → Reward/Penalty → Learn & Adapt
```

#### Key Concepts:
- **Base Price Model:** Predicts the "normal" price without surge
- **Surge Detection:** Identifies when demand exceeds supply  
- **Reinforcement Learning:** Learns optimal surge multipliers through trial and error
- **Reward Engineering:** Balances multiple business objectives

---

## 📊 What We Built

### 1. Data Processing Pipeline
- Cleaned **693,071** ride records
- Merged weather data
- Reverse-engineered Uber's surge multipliers
- Created **20+** engineered features

### 2. Base Price Prediction
- **Random Forest** model for non-surge prices
- **RMSE:** ~$1.42
- Considers distance, service type, time, location

### 3. Reinforcement Learning Agent
- **Algorithm:** Proximal Policy Optimization (PPO)
- **State Space:** 7 features (time, demand, supply, etc.)
- **Action Space:** 7 surge levels `[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]`
- **Training:** 100,000 timesteps

### 4. Intelligent Pricing Engine
The RL agent learned to:
- ✅ Surge during high demand (rush hours, bad weather)
- ✅ Avoid excessive surge that drives customers away
- ✅ Balance revenue with acceptance rates
- ✅ Respond to market conditions dynamically

---

## 🛠️ Technical Implementation

### Technologies Used
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

### Architecture
```
Data Processing → Feature Engineering → Base Model → RL Training → API → Web Interface
```

---

## Installation

### Clone the repository
```bash
git clone https://github.com/yourusername/dynamic-pricing-rl.git
cd dynamic-pricing-rl
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Place your data files
```
data/raw/
├── cab_rides.csv
└── weather.csv
```

---

## Running the System

### 1. Train all models
```bash
python scripts/run_full_pipeline.py
```

This will:
- ✅ Process and clean the data
- ✅ Engineer features
- ✅ Reverse-engineer Uber surge patterns
- ✅ Train the base price model
- ✅ Train the RL agent for 100k steps
- ✅ Save all models

**Expected output:**
```
=== UBER DYNAMIC PRICING - FULL PIPELINE ===

Step 1: Loading data...
Rides data: (693071, 10)
Weather data: (6276, 8)

Step 2: Creating temporal features...

Step 3: Merging with weather data...
Weather match rate: 99.8%

Step 4: Reverse-engineering Uber surge...

Step 5: Creating final features...

Step 6: Training base price model...
Base Price Model RMSE: $2.45

Step 7: Training RL agent...
Training for 100,000 timesteps...
[Training progress bars]

=== PIPELINE COMPLETE ===
```

### 2. Start the API server
```bash
cd api
python app.py
```
The API will be available at `http://localhost:8000`

### 3. Open the web interface
Simply open the HTML file in your browser:
```bash
open frontend/index.html
```

---

## 💡 How It Works

### The Reinforcement Learning Process

1. **State:** The agent observes current market conditions (hour, demand level, supply level, etc.)
2. **Action:** Chooses a surge multiplier from `[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]`
3. **Reward:** Receives feedback based on:
   - Revenue generated
   - Customer acceptance rate
   - Driver utilization
   - Market efficiency
4. **Learning:** Updates its policy to maximize long-term rewards

### The Reward Function
```python
reward = revenue * 0.5         # Revenue matters
       + utilization * 2       # Keep drivers busy
       - rejection_penalty     # Don't lose customers
       - high_surge_penalty    # Avoid PR disasters
```

### Using the Web Interface
1. **Select trip details:** Distance, pickup/dropoff locations, service type
2. **Adjust market conditions:** Use sliders to simulate different demand/supply scenarios
3. **Get prediction:** The RL agent recommends optimal surge pricing
4. **See the breakdown:** Base price, surge multiplier, and final price
5. **Receive recommendations:** The system advises whether it's a good time to ride

---

## What Makes Our RL Agent Smart

The agent learned several intelligent behaviors:

- ⏰ **Time Awareness:** Higher surge during rush hours (7-9am, 5-7pm) and late nights
- 🌧️ **Weather Response:** Increases surge during rain (from weather data)
- 🚗 **Supply Sensitivity:** When drivers are scarce, gradually increases surge to attract more
- 📊 **Demand Elasticity:** Learned that surge >2.0x significantly reduces acceptance
- ⚖️ **Market Balance:** Aims for ~1.5x surge as optimal balance

---

## 🔬 Technical Deep Dive

### Why PPO (Proximal Policy Optimization)?
- ✅ **Stable learning** (won't suddenly change strategy)
- ✅ **Good for continuous control**
- ✅ **Handles our multi-objective reward well**
- ✅ **Used by OpenAI for game-playing AI**

---

## Limitations & Future Work

### Current Limitations
- ❌ No real-time supply/demand data (using proxies)
- ❌ Limited to Boston area
- ❌ Weather data is sparse (16% coverage for rain)
- ❌ Doesn't consider special events

### Future Improvements
- Real-time data integration
- Multi-city expansion
- Competitor pricing consideration
- Special event detection
- Driver behavior modeling
