from train import one_test_debias_lasso
from utils import *

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import bernoulli

np.random.seed(0)

import argparse
parser = argparse.ArgumentParser(description="Setting of Experiments")

parser.add_argument('--n', type=int, default=400, help='Number of samples')
parser.add_argument('--dim', type=int, default=20, help='Dimension of multivariate Gaussian')
parser.add_argument('--testnum', type=int, default=1000, help='Number of simulations')
parser.add_argument('--k', type=int, default=2, help='Dimension that debiases')
parser.add_argument('--j', type=float, default=2.0, help='Controllable regularization parameter for lasso')
parser.add_argument('--type', type=str, default='rand inv cov', help='Type of experiment, e.g. sparse mean')

args = parser.parse_args()

def main():

    #define mean and covariance
    if args.type == 'sparse mean':
        para_dim = args.dim

        mean = np.ones(args.dim, dtype=np.float64)
        mean[5:] = 0

        inv_cov = np.zeros((args.dim,args.dim), dtype=np.float64)
        base_cov = np.zeros((args.dim,args.dim), dtype=np.float64)
        np.fill_diagonal(base_cov, 1)

    elif args.type == 'rand inv cov':
        mean = np.zeros(args.dim, dtype=np.float64)
        inv_cov = np.zeros((args.dim,args.dim), dtype=np.float64)

        for i in range(args.dim):
            for j in range(args.dim):
                element = bernoulli.rvs(0.2)
                inv_cov[i,j] = element
                inv_cov[j,i] = element
        np.fill_diagonal(inv_cov, 0)
        inv_cov[0,1] = 0
        inv_cov[1,0] = 0
        base_cov = np.random.rand(args.dim, args.dim)*0.1 # Fix dimension input
        base_cov = np.dot(base_cov, base_cov.T)        # Symmetrize the matrix
    
        np.fill_diagonal(base_cov, 12)     # Set diagonal elements to 10
        para_dim =int((args.dim*(args.dim+1))/2)

        #para_dim = int(int(np.count_nonzero(inv_cov))/2)+1

    elif args.type == 'deter inv cov':
        mean = np.zeros(args.dim, dtype=np.float64)

        inv_cov = np.zeros((args.dim,args.dim), dtype=np.float64)

        for i in range(args.dim-15):
            inv_cov[0,i] = 1
            inv_cov[i,0] = 1
        for i in range(args.dim-1):
            inv_cov[i,i+1] = 1
            inv_cov[i+1,i] = 1

        np.fill_diagonal(inv_cov, 0)

        inv_cov[0,1] = 0
        inv_cov[1,0] = 0
        print('Precision Matrix:',inv_cov)

        base_cov = np.random.rand(args.dim, args.dim)*0.1 # Fix dimension input
        base_cov = np.dot(base_cov, base_cov.T)        # Symmetrize the matrix
        np.fill_diagonal(base_cov, 12)     # Set diagonal elements to 10

        para_dim =int((args.dim*(args.dim+1))/2)
        #para_dim = int(int(np.count_nonzero(inv_cov))/2)

    l1_penalty = np.sqrt(args.j * np.log(para_dim)/args.n)
    #l1_penalty = 0
    print(f'lasso regularization parameter:{l1_penalty}')

    h=2
    l1_penalty_w = np.sqrt(h*np.log(para_dim)/args.n)

    print(f'Dimension: {para_dim}')

    #define initial guess        
    alpha_init = np.random.rand(para_dim)
    w_init = np.random.rand(para_dim)

    l = []
    z_0975 = norm.ppf(0.975)
    #critical_value = chi2.ppf(0.95, 1)
    for test in range(args.testnum):
        print(f'----------round{test+1}-----------')
        debias_lasso = one_test_debias_lasso(args.n, args.dim, mean, inv_cov, l1_penalty_w, l1_penalty, alpha_init, base_cov,w_init,args.k,cov_est=True,oracle=False)
        print(f"Standardlized Debiased Lasso result of test {test + 1}:\n", debias_lasso)
        l.append(debias_lasso)

    count_within_interval = np.sum((l>-z_0975) & (l < z_0975))

    plot_histqqplot(l,dim=args.dim,j=args.j,h=h)

    print('mean:', sum(l) / len(l),count_within_interval)

if __name__ == '__main__':
    print(args)
    main()