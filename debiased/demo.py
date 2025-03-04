from gen_data import SimuGauss_Data
from utils import get_mean,get_real_mean,get_var,plot_histqqplot
from get_inverse import get_inverse
from train import get_lasso_estimator

import numpy as np

def f(X):
    
    fX = np.einsum('...i,...j->...ij', X, X)  
    
    # Flatten each matrix in the batch
    return fX.reshape(X.shape[0], -1)

def get_lasso(n, dim, mean,inv_cov, l1_pen, alpha_init,track=False,cov_est=False):

    # Generate data
    data_gen = SimuGauss_Data(n = n, dim = dim, mean = mean, inv_cov = inv_cov)
    t_series, X, G, dG = data_gen.gen_data()
    X0 = get_mean(X,t_series)

    if cov_est:
        X = f(X)

    # Get Lasso estimator
    alpha_estimated = get_lasso_estimator(alpha_init=alpha_init, X=X, G=G, dG=dG, X0=X0, l1_penalty=l1_pen)
    
   # alpha_estimated = coordinate_descent_with_soft_thresholding(alpha_init=alpha_init, X=X, G=G, dG=dG, X0=X0, l1_penalty=l1_pen)
    if track:
        print("Estimated lasso alpha:\n", alpha_estimated)

    return alpha_estimated, X0, G, dG, X, t_series

def one_test_debias_lasso(n, dim, mean,inv_cov, l1_pen_w, l1_pen, alpha_init, w_init, k=1,cov_est=False):

    lasso, X0, G, dG, X, t_series = get_lasso(n, dim, mean = mean,
                                            inv_cov = inv_cov,
                                            l1_pen=l1_pen,
                                            alpha_init=alpha_init,
                                            cov_est=cov_est)
    
    nab1 = (np.matmul(np.matmul(np.matmul(lasso, X0.T), G), X0) + np.matmul(np.ones(n),np.matmul(dG, X))) / n
    nab2 = np.matmul(np.matmul(X0.T, G), X0) / n
    print(X)
    w = get_inverse(w_init, l1_pen_w, nab2, k)
    print(w)
    debias_lasso = lasso[k-1] - np.matmul(w, nab1)
    
    cov = get_var(lasso, t_series, X, G, dG)
    print(np.trace(cov))
    var = w.T @ cov @ w

    return ((n**0.5)*(debias_lasso-mean[k-1]))/(var**0.5)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from scipy.stats import norm

    # Parameters
    n = 50  # Number of samples
    dim = 100  # Dimensionality
    j = 2
    l1_penalty = np.sqrt(j * np.log(dim)/n)  # L1 penalty for Lasso
    print(l1_penalty)
    h=1
    l1_penalty_w = np.sqrt(h*np.log(dim)/n)
    print(l1_penalty_w)

    mean = np.ones(dim, dtype=np.float64)*0.1
    mean[5:] = 0
    inv_cov = np.zeros((dim,dim), dtype=np.float64)
    alpha_true = np.ones(dim, dtype=np.float64)*0.2
    alpha_init = np.random.rand(dim)
    w_init = np.random.rand(dim)

    test_num = 1000

    k=1
    l = []

    for test in range(test_num):
        print(f'----------round{test+1}-----------')
        debias_lasso = one_test_debias_lasso(n, dim, mean, inv_cov, l1_penalty_w, l1_penalty, alpha_init, w_init,k,cov_est=False)
        print(f"Standardlized Debiased Lasso result for test {test + 1}:\n", debias_lasso)
        l.append(debias_lasso)

    plot_histqqplot(l,dim=dim,j=j,h=h)

    print('mean:', sum(l) / len(l))

 
