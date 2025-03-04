import os
import numpy as np
import gc 
from tqdm import tqdm 
from scipy.stats import linregress
import pandas as pd 

import rpy2.robjects as robjects
from rpy2.robjects import r, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

numpy2ri.activate()

def generate_data(n, d, edge_weight=1, p=0.2):
    t_values = np.linspace(0, 1, n)
    mu = np.zeros(d)
    X_list = []
    inv_cov_base = np.zeros((d, d))
    true_changing_edges = set()
    true_nonchanging_edges = set()


    for i in range(d):
        for j in range(i + 1, d):
            if (i, j) == (0, 1):
                # Edge (0, 1) is always zero and non-changing
                inv_cov_base[i, j] = 0
                inv_cov_base[j, i] = 0
                true_nonchanging_edges.add((i, j))
            else:
                element = np.random.binomial(1, p)
                if element != 0:
                    inv_cov_base[i, j] = edge_weight
                    inv_cov_base[j, i] = edge_weight
                    true_changing_edges.add((i, j))
                else:
                    true_nonchanging_edges.add((i, j))

    for idx, t in enumerate(t_values):
        inv_cov = inv_cov_base.copy()
        for i, j in true_changing_edges:
            inv_cov[i, j] *= t 
            inv_cov[j, i] = inv_cov[i, j]

        base_cov = np.zeros((d, d))
        np.fill_diagonal(base_cov, 12)

        Theta = inv_cov + base_cov

        eigvals = np.linalg.eigvalsh(Theta)
        min_eigval = np.min(eigvals)
        if min_eigval <= 0:
            Theta += np.eye(d) * (-min_eigval + 1e-6)

        x_t = np.random.multivariate_normal(mu, np.linalg.inv(Theta))
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

def compute_permutation_thresholds(omega_array, X, positions, edge=(0,1), N_permutations=100, alpha=0.05,
                                   h=0.2, d_val=0.2, lambda_val=0.15):

    n, d, _ = omega_array.shape
    time_points = np.linspace(0, 1, n)
    i, j = edge

    permutation_slopes = []

    for perm in tqdm(range(N_permutations), desc="Computing Permutation Thresholds"):

        permuted_indices = np.random.permutation(n)
        positions_shuffled = positions[permuted_indices]

        omega_array_perm, _ = run_loggle_in_r(X, h=h, d_val=d_val, lambda_val=lambda_val, positions=positions_shuffled)

        if omega_array_perm is None:
            continue

        edge_values_perm = omega_array_perm[:, i, j]
        slope_perm, intercept_perm, _, _, _ = linregress(time_points, edge_values_perm)
        permutation_slopes.append(slope_perm)

        del omega_array_perm
        gc.collect()

    if len(permutation_slopes) < N_permutations:
        print("Not enough permutations completed. Adjust N_permutations or check for issues.")
        return None

    lower_threshold = np.percentile(permutation_slopes, alpha / 2 * 100)
    upper_threshold = np.percentile(permutation_slopes, (1 - alpha / 2) * 100)

    return lower_threshold, upper_threshold

if __name__ == "__main__":
    n = 400 
    d = 20  
    num_runs = 100 
    h = 0.2
    d_val = 0.2
    lambda_val = 0.15
    N_permutations = 100  
    alpha = 0.05  

    edge_weight = 1  

    p = 0.2

    save_directory = r"the\path\for\saving\file"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    false_positive_count = 0  
    total_runs = 0 

    print(f"\nRunning Type I error tests for edge (0, 1)...")

    X_perm, _, _ = generate_data(n, d, edge_weight=edge_weight, p=p)

    positions = np.arange(1, n + 1)

    omega_array_perm, _ = run_loggle_in_r(X_perm, h=h, d_val=d_val, lambda_val=lambda_val, positions=positions)
    if omega_array_perm is None:
        print(f"Failed to compute omega_array for permutations. Exiting...")
        exit()

    thresholds = compute_permutation_thresholds(
        omega_array_perm, X_perm, positions, edge=(0,1), N_permutations=N_permutations, alpha=alpha,
        h=h, d_val=d_val, lambda_val=lambda_val
    )

    if thresholds is None:
        print(f"Failed to compute permutation thresholds. Exiting...")
        exit()

    lower_threshold, upper_threshold = thresholds

    for run in tqdm(range(num_runs), desc="Simulations for Type I Error"):
        X, true_changing_edges, true_nonchanging_edges = generate_data(n, d, edge_weight=edge_weight, p=p)

        edge = (0, 1)
        is_edge_truly_changing = edge in true_changing_edges

        assert not is_edge_truly_changing, "Edge (0, 1) should be non-changing in this simulation."

        omega_array, positions = run_loggle_in_r(X, h=h, d_val=d_val, lambda_val=lambda_val)

        if omega_array is None:
            continue

        total_runs += 1

        n_samples = omega_array.shape[0]
        time_points = np.linspace(0, 1, n_samples)
        edge_values = omega_array[:, edge[0], edge[1]]
        observed_slope, _, _, _, _ = linregress(time_points, edge_values)

        if observed_slope < lower_threshold or observed_slope > upper_threshold:
            is_edge_detected_as_changing = True
        else:
            is_edge_detected_as_changing = False

        if is_edge_detected_as_changing:
            false_positive_count += 1  

        del X, omega_array
        gc.collect()

    if total_runs > 0:
        type_i_error_rate = false_positive_count / total_runs
        print(f"Type I Error Rate for edge (0, 1): {type_i_error_rate}")
    else:
        print("No valid runs completed.")
        type_i_error_rate = None

    results = [{'edge': (0, 1), 'type_i_error_rate': type_i_error_rate}]
    df = pd.DataFrame(results)
    filename = "type_i_error_random_graph.csv"
    filepath = os.path.join(save_directory, filename)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
