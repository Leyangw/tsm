import numpy as np

def get_obj(alpha, X, G, dG, X0):
    alpha = np.array(alpha)
    X = np.array(X)
    G = np.array(G)
    dG = np.array(dG)
    X0 = np.array(X0)

    # First part: alpha^T * (X0^T * diag(G) * X0) * alpha
    nab2 = X0.T @ G @ X0
    part1 = alpha.T @ nab2 @ alpha

    # Second part: sum(dG * (X @ alpha))
    part2 = 2 * np.sum(dG * (X @ alpha))

    # Combine both parts
    result = part1+ part2

    return result.squeeze() / (2 * G.shape[0])

def get_lasso_obj(alpha, X, G, dG, X0,l1_penalty):
    obj_val = get_obj(alpha, X, G, dG, X0)
    #k = X.shape[1]
    lasso_penalty = l1_penalty * np.sum(np.abs(alpha))

    return (obj_val + lasso_penalty).squeeze()  # Ensure the result is a scalar

def get_grad(alpha, X, G, dG, X0):

    n = X.shape[0]
    grad = (np.matmul(np.matmul(np.matmul(alpha, X0.T), G), X0) + np.matmul(np.ones(n),np.matmul(dG, X))) / n

    return grad

def get_lasso_gradient(alpha, X, G, dG, X0,l1_penalty):
    n = X.shape[0]
    grad = (np.matmul(np.matmul(np.matmul(alpha, X0.T), G), X0) + np.matmul(np.ones(n),np.matmul(dG, X))) / n

    lasso_grad = l1_penalty*np.sign(alpha)

    return grad+lasso_grad
    

def inference(alpha, X0):
    """
    Performs the inference operation alpha^T * X0.

    Parameters:
    - alpha: numpy array, vector of dimension k
    - X0: numpy array, matrix of dimension n x k

    Returns:
    - result: numpy array, result of the operation alpha^T * X0
    """
    # Perform the matrix-vector multiplication
    result = X0 @ alpha
    return result
