import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.model import compute_rmse


def run_sanity_checks(df: pd.DataFrame):

    print("\n" + "=" * 60)
    print("OPTION PRICING SANITY CHECK REPORT")
    print("=" * 60)

    # ==========================================================
    # Basic Statistics
    # ==========================================================

    print("\nMARKET MID STATISTICS")
    print(df["market_mid"].describe())

    print("\nBS(HV) STATISTICS")
    print(df["bs_baseline_price"].describe())

    print("\nBS(IV) STATISTICS")
    print(df["bs_iv_price"].describe())

    # ==========================================================
    # Relative Error Metrics
    # ==========================================================

    market_mean = df["market_mid"].mean()

    bs_hv_rmse = compute_rmse(
        df["market_mid"].values,
        df["bs_baseline_price"].values
    )

    bs_iv_rmse = compute_rmse(
        df["market_mid"].values,
        df["bs_iv_price"].values
    )

    print("\nRMSE CHECKS")
    print(f"Mean Market Price : {market_mean:,.2f} USD")
    print(f"BS(HV) RMSE       : {bs_hv_rmse:,.2f} USD")
    print(f"BS(IV) RMSE       : {bs_iv_rmse:,.2f} USD")

    print("\nNORMALIZED RMSE")

    print(
        f"BS(HV) NRMSE : "
        f"{100 * bs_hv_rmse / market_mean:.2f}%"
    )

    print(
        f"BS(IV) NRMSE : "
        f"{100 * bs_iv_rmse / market_mean:.2f}%"
    )

    # ==========================================================
    # Correlations
    # ==========================================================

    corr_hv = np.corrcoef(
        df["market_mid"],
        df["bs_baseline_price"]
    )[0, 1]

    corr_iv = np.corrcoef(
        df["market_mid"],
        df["bs_iv_price"]
    )[0, 1]

    print("\nCORRELATIONS")

    print(f"Market vs BS(HV): {corr_hv:.4f}")
    print(f"Market vs BS(IV): {corr_iv:.4f}")

    # ==========================================================
    # Top Errors
    # ==========================================================

    tmp = df.copy()

    tmp["hv_abs_error"] = np.abs(
        tmp["market_mid"] -
        tmp["bs_baseline_price"]
    )

    tmp["iv_abs_error"] = np.abs(
        tmp["market_mid"] -
        tmp["bs_iv_price"]
    )

    print("\nTOP 10 BS(HV) ERRORS")

    cols = [
        "underlying_price",
        "strike",
        "T",
        "market_mid",
        "bs_baseline_price",
        "hv_abs_error"
    ]

    print(
        tmp.nlargest(10, "hv_abs_error")[cols]
    )

    print("\nTOP 10 BS(IV) ERRORS")

    cols = [
        "underlying_price",
        "strike",
        "T",
        "market_mid",
        "bs_iv_price",
        "iv_abs_error"
    ]

    print(
        tmp.nlargest(10, "iv_abs_error")[cols]
    )

    # ==========================================================
    # Scatter Plot
    # ==========================================================

    sample = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    plt.figure(figsize=(8, 8))

    plt.scatter(
        sample["market_mid"],
        sample["bs_iv_price"],
        alpha=0.3
    )

    max_val = max(
        sample["market_mid"].max(),
        sample["bs_iv_price"].max()
    )

    plt.plot(
        [0, max_val],
        [0, max_val]
    )

    plt.xlabel("Market Mid")
    plt.ylabel("BS(IV)")
    plt.title("Market Price vs BS(IV)")
    plt.grid(True)
    plt.show()

    # ==========================================================
    # Error Histogram
    # ==========================================================

    errors = (
        df["market_mid"] -
        df["bs_iv_price"]
    )

    plt.figure(figsize=(8, 5))

    plt.hist(
        errors,
        bins=100
    )

    plt.title("BS(IV) Pricing Error Distribution")
    plt.xlabel("Error (USD)")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.show()

    print("\n" + "=" * 60)
    print("SANITY CHECKS COMPLETED")
    print("=" * 60)