import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from src import config
from src.data_fetcher import fetch_historical_deribit_data
from src.data_loader import clean_deribit_options_data, save_processed_data
from src.features import engineer_features
from src.baseline import vectorized_black_scholes
from src.model import train_chronological_split, train_residual_model, compute_rmse

# Set professional plotting style for matplotlib / seaborn
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300})

print("==================================================")
print("INITIALIZING: Hybrid Options Engine & Plotting Pipeline")
print("==================================================")

target_file = "deribit_options_chain_2024-01-01_OPTIONS.csv.gz"
raw_path = os.path.join(config.RAW_DATA_DIR, target_file)

# ------------------------------------------------------------------ #
# Step 0: Data guard                                                 #
# ------------------------------------------------------------------ #
if not os.path.exists(raw_path):
    print("\n[0/6] Target dataset missing locally — fetching...")
    downloaded_files = fetch_historical_deribit_data(
        "2024-01-01",
        "2024-01-02"
    )
    target_file = os.path.basename(downloaded_files[0])
    raw_path = downloaded_files[0]
else:
    print("\n[0/6] Verified: Raw historical data file found.")

# ------------------------------------------------------------------ #
# Step 1: Load & clean                                               #
# ------------------------------------------------------------------ #
print("\n[1/6] Loading and parsing market microstructure records...")
raw_df = clean_deribit_options_data(target_file)

# --- NEW MICROSTRUCTURE FILTER ---
before_filter = len(raw_df)

# 1. Calculate the raw BTC spread
raw_df['raw_btc_spread'] = raw_df['best_ask'] - raw_df['best_bid']

# 2. Drop dummy asks (spreads wider than 0.05 BTC) and dust bids (< 0.0005 BTC)
raw_df = raw_df[
    (raw_df['raw_btc_spread'] < 0.05) & 
    (raw_df['best_bid'] > 0.0005)
].reset_index(drop=True)

print(f"  Microstructure filter removed {before_filter - len(raw_df):,} dummy quotes.")

# 3. NOW it is safe to convert to USD
raw_df['market_mid'] = ((raw_df['best_bid'] + raw_df['best_ask']) / 2.0) * raw_df['underlying_price']
raw_df['bid_ask_spread'] = raw_df['raw_btc_spread'] * raw_df['underlying_price']

# ------------------------------------------------------------------ #
# Step 2: Feature engineering                                        #
# ------------------------------------------------------------------ #
print("[2/6] Building feature matrix and analytical Greeks...")
feat_df = engineer_features(raw_df)

# DATA QUALITY FIX: Apply the MIN/MAX delta liquidity filter
before = len(feat_df)
feat_df = feat_df[
    (feat_df['delta'].abs() >= config.MIN_DELTA) &
    (feat_df['delta'].abs() <= config.MAX_DELTA)
].reset_index(drop=True)
print(f"  Delta filter removed {before - len(feat_df):,} extreme rows "
      f"({len(feat_df):,} remaining).")

# ------------------------------------------------------------------ #
# Step 3: Black-Scholes baseline + residual target                    #
# ------------------------------------------------------------------ #
print("[3/6] Evaluating analytical Black-Scholes baseline model...")

bs_prices = vectorized_black_scholes(
    S=feat_df['underlying_price'].values,
    K=feat_df['strike'].values,
    T=feat_df['T'].values,
    r=config.RISK_FREE_RATE,
    sigma=feat_df['historical_volity'].values,
    option_type="call",
)

is_put = feat_df['option_type'].str.lower() == 'put'
if is_put.any():
    put_prices = vectorized_black_scholes(
        S=feat_df.loc[is_put, 'underlying_price'].values,
        K=feat_df.loc[is_put, 'strike'].values,
        T=feat_df.loc[is_put, 'T'].values,
        r=config.RISK_FREE_RATE,
        sigma=feat_df.loc[is_put, 'historical_volity'].values,
        option_type="put",
    )
    bs_prices[is_put.values] = put_prices

feat_df['bs_baseline_price'] = bs_prices
feat_df['residual_target']   = feat_df['market_mid'] - feat_df['bs_baseline_price']

save_processed_data(feat_df, "deribit_processed_features.csv")

# ------------------------------------------------------------------ #
# Step 4: Chronological split                                        #
# ------------------------------------------------------------------ #
print("[4/6] Constructing chronological partitions...")
train_set, val_set, test_set = train_chronological_split(feat_df)

features = [
    'moneyness', 'T', 'historical_volity',
    'delta', 'gamma', 'vega', 'bid_ask_spread',
]

X_train, y_train = train_set[features], train_set['residual_target'].values
X_val,   y_val   = val_set[features],   val_set['residual_target'].values
X_test,  y_test  = test_set[features],  test_set['residual_target'].values

# ------------------------------------------------------------------ #
# Step 5: Train residual model                                       #
# ------------------------------------------------------------------ #
print("[5/6] Training Gradient Boosted Residual Regressor...")
ml_model = train_residual_model(X_train, y_train, X_val, y_val)

os.makedirs(config.MODEL_DIR, exist_ok=True)
ml_model.save_model(os.path.join(config.MODEL_DIR, "xgboost_residual_pricer.json"))

# ------------------------------------------------------------------ #
# Step 6: Performance Evaluation & Out-of-Sample Diagnostics          #
# ------------------------------------------------------------------ #
test_pred_residuals = ml_model.predict(X_test)
hybrid_prices       = test_set['bs_baseline_price'].values + test_pred_residuals

bs_rmse     = compute_rmse(test_set['market_mid'].values, test_set['bs_baseline_price'].values)
hybrid_rmse = compute_rmse(test_set['market_mid'].values, hybrid_prices)
improvement = (bs_rmse - hybrid_rmse) / bs_rmse * 100

print("\n" + "=" * 50)
print("HYBRID PRICING ENGINE COMPLETED SUCCESSFULLY")
print("=" * 50)
print(f"Out-of-Sample BS Baseline RMSE : {bs_rmse:.4f} USD")
print(f"Out-of-Sample Hybrid RMSE      : {hybrid_rmse:.4f} USD")
print(f"Structural Error Reduction     : {improvement:.2f}%")
print("==================================================")

# ------------------------------------------------------------------ #
# Step 7: Visual Diagnostic Plot Generation                          #
# ------------------------------------------------------------------ #
print("\n[6/6] Pipeline entering diagnostic visualization phase...")

# Calculate predictions over the entire dataset for plotting consistency
feat_df['predicted_residual'] = ml_model.predict(feat_df[features])
feat_df['hybrid_price'] = feat_df['bs_baseline_price'] + feat_df['predicted_residual']
feat_df['bs_error'] = feat_df['market_mid'] - feat_df['bs_baseline_price']
feat_df['hybrid_error'] = feat_df['market_mid'] - feat_df['hybrid_price']

fig_dir = os.path.join(config.ROOT_DIR, "figures")
os.makedirs(fig_dir, exist_ok=True)

# PLOT 1: Feature Importance Matrix
print(" -> Generating Plot 1: Feature Importance Matrix...")
plt.figure(figsize=(10, 6))
importance_series = pd.Series(ml_model.feature_importances_, index=features).sort_values(ascending=True)
importance_series.plot(kind='barh', color='steelblue')
plt.title("XGBoost Feature Importance (Gain Weighting)", fontsize=14, pad=15)
plt.xlabel("Relative Importance Score", fontsize=12)
plt.ylabel("Engineered Market Features", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "01_feature_importance.png"))
plt.close()

# PLOT 2: Volatility Smile Correction (Error vs Moneyness)
print(" -> Generating Plot 2: Volatility Smile Curve Correction...")
smile_df = feat_df[(feat_df['T'] < 0.1) & (feat_df['moneyness'] > 0.8) & (feat_df['moneyness'] < 1.2)]

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

# PLOT 3: Parity Curve (Actual vs. Predicted Prices)
print(" -> Generating Plot 3: Model Parity Estimation...")
plt.figure(figsize=(9, 9))

max_val = min(feat_df['market_mid'].max(), 5000) 
plt.plot([0, max_val], [0, max_val], color='black', linestyle='--', zorder=1, label="Perfect Prediction Line")

plt.scatter(
    feat_df['market_mid'], feat_df['bs_baseline_price'], 
    alpha=0.3, color='crimson', s=10, label="Pure Black-Scholes", zorder=2
)
plt.scatter(
    feat_df['market_mid'], feat_df['hybrid_price'], 
    alpha=0.3, color='mediumseagreen', s=10, label="Hybrid Engine", zorder=3
)

plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.title("Model Predictions vs. Actual Market Mid-Prices", fontsize=14, pad=15)
plt.xlabel("Actual Market Mid-Price (USD)", fontsize=12)
plt.ylabel("Predicted Option Value (USD)", fontsize=12)
plt.legend(loc='upper left', fontsize=11, frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "03_actual_vs_predicted.png"))
plt.close()

print("\n" + "=" * 50)
print(f"ALL GRAPHICS EXPORTED: High-resolution PNG files saved to {fig_dir}/")
print("==================================================")