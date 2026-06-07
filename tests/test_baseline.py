import numpy as np
import pytest
from src.baseline import vectorized_black_scholes

def test_black_scholes_call_accuracy():
    S = np.array([100.0])
    K = np.array([100.0])
    T = np.array([1.0])
    r = 0.05
    sigma = np.array([0.20])
    
    calculated_call = vectorized_black_scholes(S, K, T, r, sigma, option_type="call")[0]
    expected_call = 10.450585
    assert abs(calculated_call - expected_call) < 1e-4

def test_black_scholes_put_accuracy():
    S = np.array([100.0])
    K = np.array([100.0])
    T = np.array([1.0])
    r = 0.05
    sigma = np.array([0.20])
    
    calculated_put = vectorized_black_scholes(S, K, T, r, sigma, option_type="put")[0]
    expected_put = 5.573505
    assert abs(calculated_put - expected_put) < 1e-4