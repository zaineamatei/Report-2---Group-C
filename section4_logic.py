import numpy as np
import pandas as pd

def simulate_future_pathways(model_result, n_years=10, n_simulations=10):
    """
    (1) What it does: Simulates multiple independent synthetic future pathways 
        from a fitted ARIMA/ARMA/AR results object using its native innovations.
    (2) Input: model_result (ARIMAResults), n_years (int), n_simulations (int).
    (3) Outputs: 2D numpy array of shape (n_steps, n_simulations)
    """
    n_steps = n_years * 12
    sim_matrix = np.zeros((n_steps, n_simulations))
    
    for i in range(n_simulations):
        # Anchor anchor_seed ensures variance reflects model innovations safely
        sim_path = model_result.simulate(n_steps, anchor='end')
        sim_matrix[:, i] = sim_path.values if hasattr(sim_path, 'values') else sim_path
        
    return sim_matrix

def compute_sediment_mass_yields(q_sim, c_sim):
    """
    (1) What it does: Computes the instant sediment mass matrix M(t) = C(t) * Q(t) [kg/s]
        and aggregates them into average monthly (1-12) and yearly yield profiles.
    (2) Input: q_sim (2D array), c_sim (2D array) of matching shapes.
    (3) Outputs: dict containing the raw mass matrix, monthly averages, and yearly averages.
    """
    mass_matrix = c_sim * q_sim
    n_steps, n_sims = mass_matrix.shape
    
    future_index = pd.date_range(start="2026-01-01", periods=n_steps, freq="M")
    
    monthly_yields_list = []
    yearly_yields_list = []
    
    for i in range(n_sims):
        df_sim = pd.DataFrame({"mass": mass_matrix[:, i]}, index=future_index)
        # Monthly baseline averages (Group by calendar month 1-12)
        monthly_avg = df_sim.groupby(df_sim.index.month)["mass"].mean()
        # Annual yield sums scaled down or tracked as mean annual flux
        yearly_avg = df_sim.resample("Y")["mass"].mean()
        
        monthly_yields_list.append(monthly_avg.values)
        yearly_yields_list.append(yearly_avg.values)
        
    return {
        "raw_mass": mass_matrix,
        "monthly_averages": np.array(monthly_yields_list).T,  # Rows=12, Cols=10 sims
        "yearly_averages": np.array(yearly_yields_list).T     # Rows=10 years, Cols=10 sims
    }