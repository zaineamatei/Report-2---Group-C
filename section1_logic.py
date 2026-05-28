import pandas as pd
import numpy as np
from scipy import stats

def load_timeseries_csv(path, value_name):
    """
    (1) What it does: Loads hydrological data from CSV, parses dates, 
        and handles missing values.
    (2) Input: path (Path object/str), value_name (str) - Column name for the data.
    (3) Outputs: df (pandas.DataFrame) - Cleaned time series with DatetimeIndex.
    """
    df = pd.read_csv(path, sep=",", parse_dates=["timestamp"], na_values=["", " "])
    original_value_column = df.columns[1]
    df = df.rename(columns={original_value_column: value_name})
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    return df.set_index("timestamp").sort_index()

def fit_linear_trend(monthly_df, value_col, alpha=0.05):
    """
    (1) What it does: Tests for stationarity via linear regression. Subtracts 
        significant trends or the mean to produce a zero-mean series.
    (2) Input: monthly_df (DataFrame), value_col (str), alpha (float) - significance.
    (3) Outputs: dict - Containing trend data, corrected series (z), and stats.
    """
    series = monthly_df[value_col].dropna()
    time_years = series.index.year + (series.index.month - 0.5) / 12
    x = time_years.to_numpy()
    y = series.to_numpy()

    regression = stats.linregress(x, y)
    trend = pd.Series(regression.intercept + regression.slope * x, index=series.index)

    if regression.pvalue < alpha:
        z = series - trend
        corr = "linear trend removed"
    else:
        z = series - series.mean()
        corr = "mean removed"

    return {
        "series": series, "trend": trend, "z": z, "slope": regression.slope,
        "pvalue": regression.pvalue, "significant_trend": regression.pvalue < alpha,
        "correction": corr, "mean_z": z.mean(), "variance_z": z.var(ddof=1)
    }