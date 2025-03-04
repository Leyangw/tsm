import random
import numpy as np
from scipy.stats import norm


class SimuGauss_Data():

    def __init__(self, n, dim, mean, inv_cov,base_cov): 
        self.num = n
        self.dim = dim
        self.mean = np.array(mean, dtype=np.float64)
        self.inv_cov = np.array(inv_cov, dtype=np.float64)
        self.base_cov = base_cov

    def get_mean(self, t):
        """
        Generates a mean vector with each element being t.

        Parameters:
        - t: float, the value to use in each element

        Returns:
        - mean_vector: numpy array, the mean vector with each element as t
        """

        mean = self.mean*t
   
        return mean

    def get_cov(self,t):
        """
        Generates a covariance matrix which is a scaled identity matrix.

        Returns:
        - cov_matrix: numpy array, the covariance matrix
        """
        
        cov = self.inv_cov*t
        cov = cov + self.base_cov

        return np.linalg.inv(cov)

    def generate(self):
        """
        Generates a series of multivariate Gaussian vectors.
        """

        t_series = np.random.rand(self.num)#np.linspace(0, 1, self.num)#
        X = np.array([self.simu_mvg(self.get_mean(t), self.get_cov(t)) for t in t_series])

        return t_series, X

    def simu_mvg(self, mean, cov):
        """
        Simulates multivariate Gaussian data.
        """
        return np.random.multivariate_normal(mean, cov)

    def gen_data(self):
        t_series, X = self.generate()
       # G = np.diag(-(t_series)*(t_series-0.2))
       # dG = np.diag(-(2*t_series-0.2))
        G = np.diag(t_series - t_series ** 2)
        dG = np.diag(1 - 2 * t_series)

        return t_series, X, G, dG
    
    def gen_data_(self):
        t_series, X = self.generate()
        return t_series, X