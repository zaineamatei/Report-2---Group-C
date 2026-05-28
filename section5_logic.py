import numpy as np
import pandas as pd
import scipy.stats as stats

def evaluate_variable_dependence(task1_results):
    """
    (1) What it does: Computes Pearson and Spearman rank correlation coefficients
        between Q and C at both stations, along with cross-station correlations.
    (2) Input: task1_results (dict containing historical 'z' dataframes).
    (3) Outputs: pd.DataFrame containing correlation coefficients and p-values.
    """
    # Align all four normalized series into a single DataFrame to ensure concurrent rows
    combined_df = pd.DataFrame({
        "Q_Diep": task1_results["Q Diepoldsau"]["z"],
        "C_Diep": task1_results["SSC Diepoldsau"]["z"],
        "Q_Gis": task1_results["Q Gisingen"]["z"],
        "C_Gis": task1_results["SSC Gisingen"]["z"]
    }).dropna()
    
    records = []
    
    # Define pairs to evaluate
    pairs = [
        ("Q vs C (Diepoldsau)", "Q_Diep", "C_Diep"),
        ("Q vs C (Gisingen)", "Q_Gis", "C_Gis"),
        ("Q Diep vs Q Gis", "Q_Diep", "Q_Gis"),
        ("C Diep vs C Gis", "C_Diep", "C_Gis")
    ]
    
    for label, col1, col2 in pairs:
        x, y = combined_df[col1].values, combined_df[col2].values
        
        pearson_r, p_pearson = stats.pearsonr(x, y)
        spearman_r, p_spearman = stats.spearmanr(x, y)
        
        records.append({
            "Comparison": label,
            "Pearson r": pearson_r,
            "Pearson p-value": p_pearson,
            "Spearman ρ": spearman_r,
            "Spearman p-value": p_spearman
        })
        
    return pd.DataFrame(records).set_index("Comparison"), combined_df