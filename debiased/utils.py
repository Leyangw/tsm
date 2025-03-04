import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.stats as stats
    
class NWEstimator():
    
    def __init__(self, t_series, X, bandwidth):
        self.bandwidth = bandwidth
        self.X = X
        self.series = t_series
        self.num = len(self.X)

    def fit(self, t):
        # Vectorized computation of weights
        weights = norm.pdf(t - self.series, scale=self.bandwidth)
        weights /= weights.sum()

        # Compute weighted mean
        X_mean = np.dot(weights,self.X)

        return X_mean

def get_mean(X, t_series, bandwidth=0.05):
    # Initialize output array
    X0 = np.zeros_like(X)
    # Create NWEstimator instance
    NW = NWEstimator(t_series, X, bandwidth)

    # Apply NWEstimator to each element in t_series
    for i in range(X.shape[0]):
        X0[i] = NW.fit(t_series[i])

    return X - X0

def get_real_mean(X,t_series):
    X0 = np.zeros_like(X)

    for i in range(X.shape[0]):
        X0[i] = t_series[i]

    return X-X0

def get_var(alpha, t_series, X, G, dG):
    # Compute the number of samples
    n = X.shape[0]

    # Compute the mean of X based on the time series
    X0 = get_mean(X, t_series)

    # Initialize the gradient array
    nab1 = np.zeros_like(X)

    # Calculate the gradient for each sample
    for i in range(n):
        # Ensure the multiplication is handled correctly; assuming 'alpha' is a scalar or compatible shape
        row_i = G[i, i] * np.dot(alpha, np.dot(X0[i].T, X0[i])) + dG[i, i] * X[i]
       # print(row_i.shape)
        nab1[i] = row_i

    cov_matrix = np.cov(nab1, rowvar=False,ddof=0)
 #   cov_matrix = nab1.T@nab1/(n-1)

    return cov_matrix
    # Check if var is 0-dimensional (scalar)
   # if var.shape == ():
    #    return np.array([var/n])
    #else:
     #   return var/n

def get_nab2mean(alpha, t_series, X, G, dG):
    # Compute the number of samples
    n = X.shape[0]

    # Compute the mean of X based on the time series
    X0 = get_mean(X, t_series)

    # Initialize the gradient array
    nab2 = np.zeros_like(X)

    # Calculate the gradient for each sample
    for i in range(n):
        # Ensure the multiplication is handled correctly; assuming 'alpha' is a scalar or compatible shape
        row_i = G[i, i] * np.dot(X0[i].T, X0[i])
       # print(row_i.shape)
        nab1[i] = row_i

    mean = np.mean(nab2, rowmean=False)
 #   cov_matrix = nab1.T@nab1/(n-1)

    return mean     

def get_l1_param(j,dim,n):

    return np.sqrt(j * np.log(dim)/n)

def extract_matrix_elements(X):
    """
    Extract elements from matrix X in the order [x11, x12, x22, x13, x23, x33, ...]
    where x_ij is the element at the ith row and jth column of X.

    Parameters:
    X (numpy.ndarray): Input matrix of shape (n_rows, n_cols)

    Returns:
    numpy.ndarray: Matrix of extracted elements
    """
    # Get the number of rows and columns
    n_rows, n_cols = X.shape

    # Extract the elements in the specified order
    result = []
    for i in range(n_rows):
        for j in range(i, n_cols):
            result.append(X[i, j])

    # Convert to numpy array
    result = np.array(result)

    # Reshape the result into a matrix format
    # Adjust the reshape dimensions as needed
    num_columns = len(result)
    reshaped_result = result.reshape(1, num_columns)  # Example reshaping to 1 row

    return reshaped_result

def plot_histqqplot(l, dim, j, h, save_path='./logs'):
    l = np.array(l)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(l, bins=15, density=True, alpha=0.6, color="royalblue", edgecolor="royalblue")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # Get the current x-axis limits
    xmin, xmax = plt.xlim()

    # Generate x values for the Gaussian PDF
    x = np.linspace(xmin, xmax, 100)

    # Calculate the Gaussian PDF with mean=0 and std=1
    p = norm.pdf(x, 0, 1)

    # Plot the Gaussian PDF as a reference
    plt.plot(x, p, color='darkorange', linewidth=4.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

   # plt.title('Histogram of Data', fontsize=20)
   #s plt.xlabel('Value', fontsize=20)
    #plt.ylabel('Density', fontsize=20)

    # Plot QQ plot against Gaussian distribution
    plt.subplot(1, 2, 2)

    x = np.linspace(min(l)-0.1, max(l)+0.1, 100)

    # Calculate the y values for a line with slope 1
    y = x

    # Plot the line
    plt.plot(x, y, color='silver', linestyle='-', linewidth=6)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    res = stats.probplot(l, dist="norm", plot=None, fit=False)
    x = res[0]  # Theoretical quantiles
    y = res[1]
    plt.scatter(x, y, color='grey', marker='^',facecolor='none', edgecolor='grey', s=250, zorder=2)  # Customize color and marker 

    # Update the title for the Q-Q plot
   # plt.title(f'QQ Plot', fontsize=20)

    # Adjust layout
    plt.tight_layout()

    # Save the plot to ./logs directory
    plt.savefig(f'{save_path}/plot_dim={dim},j={j},w={h}.png', dpi=300)  # Save with high resolution (300 DPI)