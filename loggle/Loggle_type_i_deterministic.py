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

def generate_data(n, d, edge_weight=10):
    """
    Generates data where non-zero terms in inv_cov_base stand for changing edges.
    The weight of edge (0, 1) is set to `edge_weight`.
    """
    t_values = np.linspace(0, 1, n)
    mu = np.zeros(d)
    X_list = []
    inv_cov_base = np.diag([12] * d)
    true_changing_edges = set()
    true_nonchanging_edges = set()

    for i in range(d - 15):
        inv_cov_base[0, i] = 1  
        inv_cov_base[i, 0] = 1
        true_changing_edges.add((0, i))
    for i in range(d - 1):
        inv_cov_base[i, i + 1] = 1
        inv_cov_base[i + 1, i] = 1
        true_changing_edges.add((i, i + 1))

    inv_cov_base[0, 1] = edge_weight
    inv_cov_base[1, 0] = edge_weight

    if edge_weight != 0:
        true_changing_edges.add((0, 1))
    else:
        true_nonchanging_edges.add((0, 1))

    all_possible_edges = set((i, j) for i in range(d) for j in range(i + 1, d))
    true_nonchanging_edges.update(all_possible_edges - true_changing_edges)

    for idx, t in enumerate(t_values):
        inv_cov = inv_cov_base.copy()
        for i, j in true_changing_edges:
            inv_cov[i, j] *= t  
            inv_cov[j, i] = inv_cov[i, j]

        base_cov = np.random.rand(d, d) * 0.1
        base_cov = base_cov @ base_cov.T
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

def compute_permutation_thresholds(omega_array, X, edge=(0,1), N_permutations=100, alpha=0.05,
                                   h=0.2, d_val=0.2, lambda_val=0.15):
    """
    Compute permutation thresholds once for a given edge and dataset.
    """
    n, d, _ = omega_array.shape
    time_points = np.linspace(0, 1, n)
    i, j = edge

    permutation_slopes = []

    for perm in tqdm(range(N_permutations), desc="Computing Permutation Thresholds"):
        X_shuffled = X.copy()
        np.random.shuffle(X_shuffled)

        omega_array_perm, _ = run_loggle_in_r(X_shuffled, h=h, d_val=d_val, lambda_val=lambda_val)

        if omega_array_perm is None:
            continue 

        edge_values_perm = omega_array_perm[:, i, j]
        slope_perm, intercept_perm, _, _, _ = linregress(time_points, edge_values_perm)
        permutation_slopes.append(slope_perm)

        del X_shuffled, omega_array_perm
        gc.collect()

    if len(permutation_slopes) < N_permutations:
        print("Not enough permutations completed. Adjust N_permutations or check for issues.")
        return None

    lower_threshold = np.percentile(permutation_slopes, alpha / 2 * 100)
    upper_threshold = np.percentile(permutation_slopes, (1 - alpha / 2) * 100)

    return lower_threshold, upper_threshold

def run_power_test_for_edge_weights(n, d, edge_weights, num_runs=100, h=0.2, d_val=0.2, lambda_val=0.15,
                                    N_permutations=100, alpha=0.05, save_dir=None):
    results = []

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = "." 

    for edge_weight in edge_weights:
        power_count = 0  
        false_positive_count = 0 
        total_runs = 0 
        print(f"\nRunning tests for edge weight = {edge_weight}...")

        X_perm, _, _ = generate_data(n, d, edge_weight=edge_weight)

        omega_array_perm, _ = run_loggle_in_r(X_perm, h=h, d_val=d_val, lambda_val=lambda_val)
        if omega_array_perm is None:
            print(f"Failed to compute omega_array for edge weight {edge_weight}. Skipping...")
            continue

        thresholds = compute_permutation_thresholds(
            omega_array_perm, X_perm, edge=(0,1), N_permutations=N_permutations, alpha=alpha,
            h=h, d_val=d_val, lambda_val=lambda_val
        )

        if thresholds is None:
            print(f"Failed to compute permutation thresholds for edge weight {edge_weight}. Skipping...")
            continue

        lower_threshold, upper_threshold = thresholds

        for run in tqdm(range(num_runs), desc=f"Simulations for Edge Weight {edge_weight}"):
            X, true_changing_edges, true_nonchanging_edges = generate_data(n, d, edge_weight=edge_weight)

            edge = (0, 1)
            is_edge_truly_changing = edge in true_changing_edges

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

            if is_edge_truly_changing:
                # Edge is truly changing
                if is_edge_detected_as_changing:
                    power_count += 1  # True Positive
                # else: False Negative 
            else:
                # Edge is not changing
                if is_edge_detected_as_changing:
                    false_positive_count += 1  # False Positive
                # else: True Negative

            del X, omega_array
            gc.collect()

        # Calculate power or Type I error rate
        if total_runs > 0:
            if edge_weight != 0:
                # Calculate power
                power = power_count / total_runs
                results.append({'edge_weight': edge_weight, 'power': power})
                print(f"Edge Weight: {edge_weight}, Power: {power}")
            else:
                # Calculate Type I error rate (false positive rate)
                type_i_error_rate = false_positive_count / total_runs
                results.append({'edge_weight': edge_weight, 'type_i_error_rate': type_i_error_rate})
                print(f"Edge Weight: {edge_weight}, Type I Error Rate: {type_i_error_rate}")
        else:
            print(f"No valid runs completed for edge weight {edge_weight}.")
            results.append({'edge_weight': edge_weight, 'power': None})

        df = pd.DataFrame([results[-1]])
        if edge_weight != 0:
            filename = f"edge_weight={edge_weight}_power.csv"
        else:
            filename = f"edge_weight={edge_weight}_type_i_error.csv"
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")

        del X_perm, omega_array_perm
        gc.collect()

    return results


if __name__ == "__main__":
    n = 400  
    d = 20   
    num_runs = 1000 
    h = 0.2
    d_val = 0.2
    lambda_val = 0.15
    N_permutations = 100 
    alpha = 0.05  

    edge_weights = [0]

    save_directory = r"the\path\for\saving\file"

    # Run Type I error test for edge weight 0
    all_results = run_power_test_for_edge_weights(
        n, d, edge_weights, num_runs=num_runs, h=h, d_val=d_val, lambda_val=lambda_val,
        N_permutations=N_permutations, alpha=alpha, save_dir=save_directory
    )

    df_all = pd.DataFrame(all_results)
    all_results_filepath = os.path.join(save_directory, "type_i_error_results.csv")
    df_all.to_csv(all_results_filepath, index=False)
    print(f"All results saved to {all_results_filepath}")
