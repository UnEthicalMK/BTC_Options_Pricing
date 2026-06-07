import numpy as np
from scipy.stats import norm


def vectorized_black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Computes analytical European option prices using numpy vectorization.
    """
    T = np.maximum(T, 1e-5)
    sigma = np.maximum(sigma, 1e-5)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        # BUG FIX: was norm.cdf(d2), must be norm.cdf(-d2)
        # Standard BS put: K*e^(-rT)*N(-d2) - S*N(-d1)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    return price


def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Computes delta, gamma, and vega values for feature engineering.
    Gamma and vega are identical for calls and puts (same formula).
    """
    T = np.maximum(T, 1e-5)
    sigma = np.maximum(sigma, 1e-5)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * pdf_d1

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
    else:
        # put delta = N(d1) - 1  (put-call parity)
        delta = norm.cdf(d1) - 1.0

    return delta, gamma, vega