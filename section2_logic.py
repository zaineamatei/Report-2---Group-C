import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

def compute_correlation_analysis_with_gaps(series, max_lag=12):
    """
    (1) What it does: Computes ACF/PACF by averaging segments to handle data gaps 
        (e.g., Gisingen 2009/Diepoldsau 2019) per instructor suggestion[cite: 10, 15].
    (2) Input: series (pd.Series) - Zero-mean monthly data; max_lag (int).
    (3) Outputs: dict - Averaged ACF, PACF, lags, and confidence bounds.
    """
    # Identify where data exists
    mask = series.notna()
    valid_data = series[mask]
    
    if len(valid_data) == 0:
        return None

    # Identify jumps in the timeline to find segments
    # Calculate total months to identify gaps larger than 1 month
    time_index = valid_data.index
    month_diffs = (time_index.year * 12 + time_index.month)
    gap_indices = np.where(np.diff(month_diffs) > 1)[0] + 1
    segments = np.split(valid_data, gap_indices)
    
    acf_list = []
    pacf_list = []
    
    for seg in segments:
        # Segment must be long enough to compute correlation reliably
        if len(seg) > max_lag + 1:
            # We use fft=True for efficiency and 'ywm' for PACF stability
            acf_list.append(acf(seg, nlags=max_lag, fft=True))
            pacf_list.append(pacf(seg, nlags=max_lag, method="ywm"))
            
    # Safety check: if no segments were long enough, use the whole (dropped) series
    if not acf_list:
        acf_vals = acf(valid_data, nlags=max_lag, fft=True)
        pacf_vals = pacf(valid_data, nlags=max_lag, method="ywm")
    else:
        acf_vals = np.mean(acf_list, axis=0)
        pacf_vals = np.mean(pacf_list, axis=0)
        
    n = len(valid_data)
    conf_bound = 1.96 / np.sqrt(n) # Approximate 95% confidence interval 
    
    return {
        "acf": acf_vals, 
        "pacf": pacf_vals, 
        "lags": np.arange(max_lag + 1), 
        "conf_bound": conf_bound
    }

def fit_ar_arma_candidates(series, max_ar_order=12, max_arma_p=12, max_arma_q=3):
    """
    (1) What it does: Fits candidate AR and ARMA models (up to lag 12) and 
        selects the best converged models using BIC (Parsimony).
    (2) Input: series (pd.Series), max_ar_order (int), max_arma_p (int), max_arma_q (int).
    (3) Outputs: dict - best models, orders, and selection table.
    """
    z = series.dropna()
    fitted_models = []

    # Iterate through AR candidates
    for p in range(1, max_ar_order + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", (ConvergenceWarning, UserWarning))
            try:
                model = ARIMA(z, order=(p, 0, 0), trend="n").fit()
                # Check convergence explicitly
                if model.mle_retvals.get("converged", False):
                    fitted_models.append({
                        "model_type": "AR", "order": (p, 0, 0), "p": p, "q": 0,
                        "aic": model.aic, "bic": model.bic, "model": model
                    })
            except Exception:
                continue

    # Iterate through ARMA candidates (p, 0, q)
    for p in range(1, max_arma_p + 1):
        for q in range(1, max_arma_q + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", (ConvergenceWarning, UserWarning))
                try:
                    model = ARIMA(z, order=(p, 0, q), trend="n").fit()
                    if model.mle_retvals.get("converged", False):
                        fitted_models.append({
                            "model_type": "ARMA", "order": (p, 0, q), "p": p, "q": q,
                            "aic": model.aic, "bic": model.bic, "model": model
                        })
                except Exception:
                    continue

    if not fitted_models:
        return None

    selection_table = pd.DataFrame(fitted_models)

    # Filter best models based on BIC (Parsimony requirement)
    best_ar = selection_table[selection_table["model_type"] == "AR"].sort_values("bic").iloc[0]
    best_arma = selection_table[selection_table["model_type"] == "ARMA"].sort_values("bic").iloc[0]

    return {
        "best_ar_order": best_ar["order"],
        "best_ar_model": best_ar["model"],
        "best_ar_bic": best_ar["bic"],
        "best_arma_order": best_arma["order"],
        "best_arma_model": best_arma["model"],
        "best_arma_bic": best_arma["bic"],
        "selection_table": selection_table.sort_values("bic")
    }
