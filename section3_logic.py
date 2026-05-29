import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox

def compute_model_evaluation_data(series, model_result, max_lag=12):
    """
    (1) What it does: Computes the empirical ACF of the data, the exact theoretical ACF 
        implied by the fitted model coefficients, and the ACF of the model's residuals.
        Performs a Portmanteau test for residual independence and evaluates
        residual normality using the PPCC score.
    (2) Input: 
        - series (pd.Series): The zero-mean corrected time series (z).
        - model_result (ARIMAResults): The fitted statsmodels ARIMA object.
        - max_lag (int): Maximum number of lags to evaluate (default 12).
    (3) Outputs: dict containing arrays for empirical ACF, theoretical ACF, 
        residual ACF, the mathematical 95% white noise confidence threshold for residuals,
        the Portmanteau p-value, the PPCC score, and the raw residuals.
    """
    valid_data = series.dropna()
    
    # 1. Empirical ACF of observed data
    observed_acf = acf(valid_data, nlags=max_lag, fft=True)
    
    # 2. Extract coefficients directly via string matching from the parameter Series
    params = model_result.params
    ar_coefs = np.array([v for k, v in params.items() if 'ar.L' in k])
    ma_coefs = np.array([v for k, v in params.items() if 'ma.L' in k])
    
    # Format polynomials standard form: [1, -ar_1, -ar_2, ...] and [1, ma_1, ma_2, ...]
    ar_poly = np.r_[1, -ar_coefs] if len(ar_coefs) > 0 else np.array([1])
    ma_poly = np.r_[1, ma_coefs] if len(ma_coefs) > 0 else np.array([1])
    
    process = ArmaProcess(ar_poly, ma_poly)
    theoretical_acf = process.acf(lags=max_lag + 1)
    
    # 3. Residuals and Residual ACF
    residuals = model_result.resid.dropna()
    n_resid = len(residuals)
    residual_acf = acf(residuals, nlags=max_lag, fft=True)
    
    # 4. 95% White Noise Confidence Bounds for Residuals
    conf_bound_resid = 1.96 / np.sqrt(n_resid)
    
    # 5. Cumulative Portmanteau Test at 5% significance level
    df_adjust = len(ar_coefs) + len(ma_coefs)
    
    # Use 24 cumulative lags for high-order models to ensure positive degrees of freedom
    test_lags = 24 if max_lag <= df_adjust else max_lag
    
    # PASS test_lags AS AN INTEGER (NOT A LIST) FOR A CUMULATIVE TEST
    lb_df = acorr_ljungbox(residuals, lags=test_lags, model_df=df_adjust, return_df=True)
    
    # Extract the p-value at the final cumulative lag horizon
    lb_pvalue = lb_df["lb_pvalue"].iloc[-1]
    
    # 6. PPCC (Probability Plot Correlation Coefficient) Computation
    sorted_resid = np.sort(residuals)
    prob_positions = (np.arange(1, n_resid + 1) - 0.375) / (n_resid + 0.25)
    theoretical_quantiles = stats.norm.ppf(prob_positions)
    
    # Calculate the empirical PPCC score (r)
    ppcc_score, _ = stats.pearsonr(sorted_resid, theoretical_quantiles)
    
    return {
        "lags": np.arange(max_lag + 1),
        "observed_acf": observed_acf,
        "theoretical_acf": theoretical_acf[:max_lag + 1],
        "residual_acf": residual_acf,
        "conf_bound_resid": conf_bound_resid,
        "lb_pvalue": lb_pvalue,
        "ppcc_score": ppcc_score,
        "residuals": residuals
    }