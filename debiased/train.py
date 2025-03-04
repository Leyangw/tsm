from gen_data import SimuGauss_Data
from objective import get_lasso_obj,get_lasso_gradient,get_grad
from utils import *
from get_inverse import get_inverse

import numpy as np
from scipy.optimize import minimize

def f(X):
    
    def pairwise_products(x):
        d = len(x)
        result = np.zeros(d * (d + 1) // 2)
        index = 0
        for i in range(d):
            for j in range(i, d):
                result[index] = x[i] * x[j]
                index += 1
        return result

    n, d = X.shape
    result = np.zeros((n, d * (d + 1) // 2))  # Initialize result array
    
    for i in range(n):
        result[i] = pairwise_products(X[i])

    return result

def oracle_f(M, X):

    n, k = X.shape
    results = []
    for row in X:
        #row_result = [row[0]*row[1]]
        row_result = []
        for i in range(k):
            for j in range(i, k):
                if M[i, j] != 0:
                    row_result.append(row[i] * row[j])
        results.append(np.array(row_result))

    return np.array(results)

def oracle_realmean(M, base_cov, t_series):

    n = len(t_series)
    k = M.shape[0]
    results = []

    for t in t_series:
        row_result = []
        real_mean = np.linalg.inv(M*t+base_cov)
        for i in range(k):
            for j in range(i, k):
                if M[i, j] != 0:
                    row_result.append(real_mean[i,j])
        results.append(np.array(row_result))
    return np.array(results)

def get_lasso_estimator(alpha_init, X, G, dG, X0, l1_penalty, epochs=5000):
    alpha = np.array(alpha_init, dtype=np.float64)

   # obj_fun = get_lasso_obj(alpha, X, G, dG, X0,l1_penalty)

    result = minimize(fun = get_lasso_obj,
                        x0 = alpha,
                        args=(X, G, dG, X0, l1_penalty),
                        jac=get_lasso_gradient,
                        method='L-BFGS-B',
                        options={'maxiter': epochs})

    alpha = result.x
    
    return alpha

def get_lasso(n, dim, mean,inv_cov, l1_pen, alpha_init,base_cov,track=False,cov_est=True,oracle=False):

    # Generate data
    data_gen = SimuGauss_Data(n = n, dim = dim, mean = mean, inv_cov = inv_cov,base_cov=base_cov)
    t_series, X, G, dG = data_gen.gen_data()
   
    if cov_est:
        if not oracle:
            X = f(X)
        else:
            X = oracle_f(inv_cov, X)
   
    X0 = get_mean(X,t_series)
    
    # Get Lasso estimator
    alpha_estimated = get_lasso_estimator(alpha_init=alpha_init, X=X, G=G, dG=dG, X0=X0, l1_penalty=l1_pen)
    
    if track:
        print("Estimated lasso alpha:\n", alpha_estimated)

    return alpha_estimated, X0, G, dG, X, t_series

def one_test_debias_lasso(n, dim, mean,inv_cov, l1_pen_w, l1_pen, alpha_init,base_cov, w_init, k=1,cov_est=True,oracle=False):

    lasso, X0, G, dG, X, t_series = get_lasso(n, dim, mean = mean,
                                            inv_cov = inv_cov,
                                            l1_pen=l1_pen,
                                            base_cov=base_cov,
                                            alpha_init=alpha_init,
                                            cov_est=cov_est,
                                            oracle=oracle)

    true = extract_matrix_elements(inv_cov)

    if oracle:
        
        ls = lasso.copy()
        ls[0] = 0
        
        Siama_A_inv = np.linalg.inv(np.matmul(np.matmul(X0.T, G), X0))
        Sigma_B = n * get_var(ls, t_series, X, G, dG)
        Sigma = np.dot(np.dot(Siama_A_inv,Sigma_B),Siama_A_inv)
   
        return lasso[k-1]/Sigma[k-1,k-1]**0.5

    nab1 = (np.matmul(np.matmul(np.matmul(lasso, X0.T), G), X0) + np.matmul(np.ones(n),np.matmul(dG, X))) / n
    nab2 = np.matmul(np.matmul(X0.T, G), X0) / n
 
    w = get_inverse(w_init, l1_pen_w, nab2, k)
    
    debias_lasso = lasso[k-1] - np.matmul(w, nab1)
    
    cov = get_var(lasso, t_series, X, G, dG)
 
    var = w.T @ cov @ w
    
    return (((n**0.5)*(debias_lasso))/(var**0.5))



