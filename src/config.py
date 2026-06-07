import os
from pathlib import Path

# Repository Root Directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Financial & Data Filtering Constants
RISK_FREE_RATE = 0.05  # 5% constant annualised risk-free rate
MIN_DELTA = 0.10       # Drop extreme OTM options (|delta| < 0.10)
MAX_DELTA = 0.90       # Drop extreme deep-ITM options (|delta| > 0.90)

# Machine Learning Hyperparameters
# subsample / colsample_bytree add stochastic regularisation to reduce overfitting.
# reg_alpha (L1) and reg_lambda (L2) add weight regularisation.
# early_stopping_rounds halts training when val loss stops improving, preventing
# overfitting regardless of n_estimators ceiling.
XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 500,          # raised ceiling; early stopping will find the optimum
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "subsample": 0.80,            # row subsampling per tree
    "colsample_bytree": 0.80,     # feature subsampling per tree
    "reg_alpha": 0.01,            # L1 regularisation
    "reg_lambda": 1.0,            # L2 regularisation (XGBoost default is 1)
    "min_child_weight": 5,        # minimum sum of instance weight in a leaf
}

EARLY_STOPPING_ROUNDS = 30       # stop if val RMSE hasn't improved for 30 rounds