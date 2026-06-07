import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from src import config


def train_chronological_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Splits the dataframe chronologically (no shuffling — shuffling would
    allow future records to appear in training data).
    Returns train, validation, and test slices.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    print(f"  Split sizes — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")
    return train_df, val_df, test_df


def train_residual_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> XGBRegressor:
    """
    Trains an XGBoost regressor to approximate Black-Scholes pricing residuals.

    OVERFITTING FIX: early_stopping_rounds causes training to halt when
    validation RMSE has not improved for EARLY_STOPPING_ROUNDS consecutive
    boosting rounds, regardless of the n_estimators ceiling.  This makes
    the model robust to the choice of n_estimators.
    """
    model = XGBRegressor(**config.XGB_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print("  XGBoost training completed.")
    return model


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))