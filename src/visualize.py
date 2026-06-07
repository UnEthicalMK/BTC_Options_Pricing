import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from src import config

# Set professional plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300})

print("==================================================")
print("INITIALIZING: Quantitative Diagnostic Visualization")
print("==================================================")

# 1. Path Verification and Data Loading
# Note: config.PROCESSED_DATA_DIR should point to the directory where save_processed_data saves files
data_path = os.path.join(config.PROCESSED_DATA_DIR, "deribit_processed_features.csv")
model_path = os.path.join(config.MODEL_DIR, "xgboost_residual_pricer.json")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing feature matrix: {data_path}. Run main.py first.")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Missing trained model artifact: {model_path}. Run main.py first.")

print("Loading processed feature data and trained XGBoost model...")
df = pd.read_csv(data_path)

# 2. Model Rehydration and Out-of-Sample Prediction Generation
model = xgb.XGBRegressor()
model.load_model(model_path)

features = ['moneyness', 'T', 'historical_volity', 'delta', 'gamma', 'vega', 'bid_ask_spread']
X = df[features]

# Reconstruct pricing targets and residual errors
df['predicted_residual'] = model.predict(X)
df['hybrid_price'] = df['bs_baseline_price'] + df['predicted_residual']
df['bs_error'] = df['market_mid'] - df['bs_baseline_price']
df['hybrid_error'] = df['market_mid'] - df['hybrid_price']

# Initialize figure directory
fig_dir = os.path.join(config.ROOT_DIR, "figures")
os.makedirs(fig_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# Plot 1: Feature Importance Matrix                                  #
# ------------------------------------------------------------------ #
print("\n[1/3] Generating Feature Importance Plot...")
plt.figure(figsize=(10, 6))
importance_series = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
importance_series.plot(kind='barh', color='steelblue')

plt.title("XGBoost Feature Importance (Gain Weighting)", fontsize=14, pad=15)
plt.xlabel("Relative Importance Score", fontsize=12)
plt.ylabel("Engineered Market Features", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "01_feature_importance.png"))
plt.close()

# ------------------------------------------------------------------ #
# Plot 2: Volatility Smile Correction (Error vs Moneyness)           #
# ------------------------------------------------------------------ #
print("[2/3] Generating Volatility Smile Correction Plot...")
# Filter to near-term expiries (T < 0.1) and standard liquid trading bounds to clearly see the smile
smile_df = df[(df['T'] < 0.1) & (df['moneyness'] > 0.8) & (df['moneyness'] < 1.2)]

plt.figure(figsize=(12, 7))
sns.scatterplot(
    x=smile_df['moneyness'], 
    y=smile_df['bs_error'], 
    alpha=0.4, label='Black-Scholes Error', color='crimson', s=15
)
sns.scatterplot(
    x=smile_df['moneyness'], 
    y=smile_df['hybrid_error'], 
    alpha=0.4, label='Hybrid Model Error', color='mediumseagreen', s=15
)

plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.title("Pricing Error vs. Moneyness (The Volatility Smile)", fontsize=14, pad=15)
plt.xlabel("Moneyness (Spot Price / Strike Price)", fontsize=12)
plt.ylabel("Pricing Error (Market Mid - Model Price) USD", fontsize=12)
plt.legend(loc='upper right', fontsize=11, frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "02_volatility_smile_correction.png"))
plt.close()

# ------------------------------------------------------------------ #
# Plot 3: Parity Curve (Actual vs. Predicted Prices)                 #
# ------------------------------------------------------------------ #
print("[3/3] Generating Parity Prediction Plot...")
plt.figure(figsize=(9, 9))

# Establish 45-degree identity line representing perfect theoretical matching
max_val = min(df['market_mid'].max(), 5000) 
plt.plot([0, max_val], [0, max_val], color='black', linestyle='--', zorder=1, label="Perfect Prediction Line")

# Scatter analytical baseline prices vs market actuals
plt.scatter(
    df['market_mid'], df['bs_baseline_price'], 
    alpha=0.3, color='crimson', s=10, label="Pure Black-Scholes", zorder=2
)
# Scatter non-linear machine learning corrected prices vs market actuals
plt.scatter(
    df['market_mid'], df['hybrid_price'], 
    alpha=0.3, color='mediumseagreen', s=10, label="Hybrid Engine", zorder=3
)

plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.title("Model Predictions vs. Actual Market Mid-Prices", fontsize=1)