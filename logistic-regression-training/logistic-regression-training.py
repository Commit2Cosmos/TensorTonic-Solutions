import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    n_samples = X.shape[1]
    w = np.zeros(n_samples)
    b = np.zeros(1)

    for i in range(steps):
        p = _sigmoid(X @ w + b)
        loss = -1/n_samples * np.sum(y*np.log(p) + (1-y)*np.log(1-p))
        w_grad = X.T @ (p - y) / len(y)
        b_grad = np.mean(p - y)

        w = w - lr * w_grad
        b = b - lr * b_grad

    return (w, b)