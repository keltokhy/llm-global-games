import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

def compute_level_k():
    root = Path(__file__).resolve().parent.parent / "output" / "mistralai--mistral-small-creative"
    bc_path = root / "experiment_bc_sweep_summary.csv"
    if not bc_path.exists():
        print("Data not found")
        return
        
    df = pd.read_csv(bc_path)
    sigma = 0.3
    
    results = {}
    
    # We want to see how well BNE, L1, and L2 predict empirical join fractions
    # overall across the B/C sweep.
    y_emp = df["join_fraction_valid"].values
    
    # Theoretical attack masses
    theta_star = df["theta_star_target"].values
    theta = df["theta"].values
    
    # BNE
    x_bne = theta_star + sigma * norm.ppf(theta_star)
    a_bne = norm.cdf((x_bne - theta) / sigma)
    
    # L1
    x_l1 = 0.5 - sigma * norm.ppf(theta_star)
    a_l1 = norm.cdf((x_l1 - theta) / sigma)
    
    # L2
    # L2 expects others to play L1.
    # Regime falls if A_L1(theta) > theta
    # A_L1(theta) = norm.cdf((x_l1 - theta) / sigma)
    # We need the root theta_L1 such that A_L1(theta_L1) = theta_L1
    from scipy.optimize import root_scalar
    
    a_l2 = np.zeros_like(theta)
    for i, ts in enumerate(theta_star):
        xl1_val = 0.5 - sigma * norm.ppf(ts)
        def obj(t):
            return norm.cdf((xl1_val - t) / sigma) - t
        res = root_scalar(obj, bracket=[-2, 2])
        theta_L1 = res.root
        xl2_val = theta_L1 - sigma * norm.ppf(ts)
        a_l2[i] = norm.cdf((xl2_val - theta[i]) / sigma)

    
    for name, a_pred in [("BNE", a_bne), ("L1", a_l1), ("L2", a_l2)]:
        mse = np.mean((y_emp - a_pred)**2)
        rmse = np.sqrt(mse)
        r, _ = stats.pearsonr(a_pred, y_emp)
        results[name] = {"rmse": float(rmse), "r": float(r)}
        print(f"{name}: RMSE={rmse:.4f}, r={r:.4f}")
        
if __name__ == "__main__":
    compute_level_k()
