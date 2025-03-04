import numpy as np
from scipy.optimize import minimize

def inv_loss(w, lam, de, k):
    """
    Inverse loss function calculation.

    Parameters:
    - w: numpy array, vector of weights
    - lam: float, regularization parameter
    - de: numpy array, matrix for quadratic term
    - k: int, index for linear term

    Returns:
    - obj: float, the result of the inverse loss function
    """

    k -= 1
    e_k = np.zeros(de.shape[0], dtype=np.float64)
    e_k[k] = 1.0
    obj = 0.5*(w.T @ de @ w) - w.T @ e_k + lam * np.sum(np.abs(w))

    return obj

def inv_grad(w, lam, de, k):
    k -= 1
    e_k = np.zeros(de.shape[0], dtype=np.float64)
    e_k[k] = 1.0
    grad = (w.T @ de).T - e_k + lam * np.sign(w)

    return grad

def get_inverse(w_init, lam, de, k, epochs=1500):
    """
    Function to find the inverse weights using optimization.

    Parameters:
    - w_init: numpy array, initial weights
    - lam: float, regularization parameter
    - de: numpy array, matrix for quadratic term
    - k: int, index for linear term
    - lr: float, learning rate
    - epochs: int, number of epochs for optimization

    Returns:
    - w: numpy array, optimized weights
    """

    result = minimize(fun = inv_loss,
                      x0=w_init,
                      args=(lam, de, k),
                      jac= inv_grad,
                      method='L-BFGS-B',
                      options={'maxiter': epochs,
                        'ftol': 1e-9})

    return result.x

if __name__ == '__main__':
    # Example usage
    w_init = np.random.rand(10)  # Initial weights
    lam = 0.1  # Regularization parameter
    de = np.random.rand(10, 10)  # Matrix for quadratic term
    k = 1  # Index for linear term
    lr = 0.01  # Learning rate
    epochs = 1000  # Number of epochs

    # Find the inverse weights
    w_optimized = get_inverse(w_init, lam, de, k, lr, epochs)
    print("Optimized weights:\n", w_optimized)
