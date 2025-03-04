import numpy as np
import matplotlib.pyplot as plt
import gc  
from tqdm import tqdm 
from scipy.stats import linregress

import rpy2.robjects as robjects
from rpy2.robjects import r, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter


numpy2ri.activate()

def generate_data(n, d):
    # Set random seed for reproducibility
    # np.random.seed(2)
    
    G = (np.random.rand(d, d) + np.eye(d)) < 0.023
    G = np.triu(G, 1)
    G = G + G.T
    G = G.astype(int)
    
    np.fill_diagonal(G, 0)
    
    true_changing_edges = set()
    true_nonchanging_edges = set()
    for i in range(d):
        for j in range(i + 1, d):
            if G[i, j] == 1:
                true_changing_edges.add((i, j))
            else:
                true_nonchanging_edges.add((i, j))
    
    t_values = np.random.rand(n)
    
    X_list = []
    
    base = np.random.randn(d, 2 * d)
    base = base @ base.T / (2 * d)
    
    for idx, t in enumerate(t_values):
        mu = np.zeros(d)

        Theta = np.eye(d) + base
        np.fill_diagonal(Theta, 2.0)
        
        Theta_changing = np.zeros_like(Theta)
        Theta_changing[G == 1] = 0.45 * t
        Theta += Theta_changing
        
        Theta = (Theta + Theta.T) / 2
        
        eigvals = np.linalg.eigvalsh(Theta)
        min_eigval = np.min(eigvals)
        if min_eigval <= 0:
            Theta += np.eye(d) * (-min_eigval + 1e-6)
        
        Cov = np.linalg.inv(Theta)
        
        x_t = np.random.multivariate_normal(mu, Cov)
        
        X_list.append(x_t)

    X = np.vstack(X_list)
    
    return X, true_changing_edges, true_nonchanging_edges

def run_loggle_in_r(X, h=0.2, d_val=0.2, lambda_val=0.15, positions=None):

    loggle = importr('loggle')

    X_np = X.T

    with localconverter(robjects.default_converter + numpy2ri.converter):
        X_r = robjects.conversion.py2rpy(X_np)

    robjects.globalenv['X'] = X_r

    if positions is None:
        n = X.shape[0]
        positions = np.arange(1, n + 1)

    with localconverter(robjects.default_converter + numpy2ri.converter):
        pos_r = robjects.conversion.py2rpy(np.array(positions))

    robjects.globalenv['pos'] = pos_r

    r_code = f"""
    library(loggle)
    result <- loggle(X, pos = pos, h = {h}, d = {d_val}, lambda = {lambda_val},
                     fit.type = "pseudo", refit = TRUE, num.thread = 1)
    omega_list <- result$Omega
    """

    try:
        robjects.r(r_code)
    except Exception as e:
        print(f"Error in executing R code: {e}")
        return None, None

    omega_list_r = robjects.globalenv['omega_list']

    omega_list_py = []

    for omega_r in omega_list_r:
        with localconverter(robjects.default_converter + numpy2ri.converter):
            omega_dense = np.array(robjects.r['as.matrix'](omega_r))
            omega_list_py.append(omega_dense)

    omega_array = np.array(omega_list_py)

    robjects.r('rm(list = ls())')

    return omega_array, positions

def compute_edge_slopes(omega_array):
    n, d, _ = omega_array.shape
    time_points = np.linspace(0, 1, n)
    observed_slopes = {}
    for i in range(d):
        for j in range(i + 1, d):
            edge_values = omega_array[:, i, j]
            slope, _, _, _, _ = linregress(time_points, edge_values)
            observed_slopes[(i, j)] = slope
    return observed_slopes

def compute_tpr_fpr(observed_slopes, true_changing_edges, true_nonchanging_edges, thresholds):
    tpr_list = []
    fpr_list = []
    for threshold in thresholds:
        estimated_changing_edges = set()
        for edge, slope in observed_slopes.items():
            if abs(slope) >= threshold:
                estimated_changing_edges.add(edge)
        # Compute TP, FP, FN, TN
        TP = len(estimated_changing_edges & true_changing_edges)
        FP = len(estimated_changing_edges & true_nonchanging_edges)
        FN = len(true_changing_edges - estimated_changing_edges)
        TN = len(true_nonchanging_edges - estimated_changing_edges)
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        tpr_list.append(TPR)
        fpr_list.append(FPR)
    return tpr_list, fpr_list

def run_simulation(n, d, num_runs=3, h=0.2, d_val=0.2, lambda_val=0.15):
    thresholds = np.linspace(0, 0.7, 20)
    tpr_lists = []
    fpr_lists = []

    for run in tqdm(range(num_runs), desc="Running Simulations"):
        X, true_changing_edges, true_nonchanging_edges = generate_data(n, d)

        omega_array, positions = run_loggle_in_r(X, h=h, d_val=d_val, lambda_val=lambda_val)

        if omega_array is None:
            continue

        observed_slopes = compute_edge_slopes(omega_array)

        tpr_list, fpr_list = compute_tpr_fpr(
            observed_slopes, true_changing_edges, true_nonchanging_edges, thresholds
        )

        tpr_lists.append(tpr_list)
        fpr_lists.append(fpr_list)

        # Clean up memory
        del X, omega_array
        gc.collect()


    tpr_array = np.array(tpr_lists) 
    fpr_array = np.array(fpr_lists) 

    avg_tpr = np.mean(tpr_array, axis=0) 
    avg_fpr = np.mean(fpr_array, axis=0) 

    return thresholds, avg_tpr.tolist(), avg_fpr.tolist()

import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='o', linestyle='-', color='b')

    for i, threshold in enumerate(thresholds):
        plt.annotate(f'{threshold:.2f}', (fpr[i], tpr[i]))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    n = 1000  
    d = 40   
    num_runs = 10  
    h = 0.2
    d_val = 0.2
    lambda_val = 0.15

    print("Running simulations with modified data generation...")
    thresholds, avg_tpr, avg_fpr = run_simulation(
        n, d, num_runs=num_runs, h=h, d_val=d_val, lambda_val=lambda_val
    )

    print(f"Thresholds: {thresholds}")
    print(f"Average True Positive Rates: {avg_tpr}")
    print(f"Average False Positive Rates: {avg_fpr}")


    plot_roc_curve(avg_fpr, avg_tpr, thresholds)
    import pandas as pd

    roc_data = pd.DataFrame({
        'Threshold': thresholds,
        'Avg_TPR': avg_tpr,
        'Avg_FPR': avg_fpr
    })

    roc_data.to_csv('roc_data.csv', index=False)

    print("ROC data saved to 'roc_data.csv'.")

    roc_data.to_csv("the\path\for\saving\file" , index=False)
