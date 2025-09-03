import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kmeans import kmeans
from sklearn.metrics import mean_squared_error


def gaussian_design_matrix(x: np.ndarray, mu: np.ndarray, s: float) -> np.ndarray:
    """Construct the Gaussian basis design matrix Φ for input x."""
    x = x.flatten()   # ensure 1D
    mu = mu.flatten() # ensure 1D
    
    N = len(x)
    M = len(mu)
    phi = np.ones((N, M + 1))  # +1 for bias term
    
    for j in range(M):
        phi[:, j + 1] = np.exp(-0.5 * ((x - mu[j]) / s) ** 2)
    return phi



def fit_ridge_regression_gaussian(x: np.ndarray, y: np.ndarray, mu: np.ndarray, s: float, lam: float) -> np.ndarray:
    """Fit ridge regression with Gaussian basis functions."""
    phi = gaussian_design_matrix(x, mu, s)
    identity = np.identity(phi.shape[1])
    return np.linalg.inv(lam * identity + phi.T @ phi) @ phi.T @ y


def predict_gaussian(x: np.ndarray, w: np.ndarray, mu: np.ndarray, s: float) -> np.ndarray:
    """Predict outputs using Gaussian basis functions."""
    phi = gaussian_design_matrix(x, mu, s)
    return phi @ w


def plot_fit_gaussian(x: np.ndarray, y: np.ndarray, w: np.ndarray, mu: np.ndarray, s: float, lam: float) -> None:
    """Plot training data and fitted Gaussian basis curve."""
    x_range = np.linspace(np.min(x), np.max(x), 200)
    y_pred = predict_gaussian(x_range, w, mu, s)

    plt.scatter(x, y, color="blue", label="Data")
    plt.plot(x_range, y_pred, color="red", label="Gaussian Basis Model")
    plt.title(f"Gaussian Basis Regression (M={len(mu)}, σ={s}, λ={lam}, N={len(x)})")
    plt.legend()
    plt.show()


def solve_gaussian(x: np.ndarray, y: np.ndarray, mu: np.ndarray, s: float, lam: float) -> None:
    """Solve ridge regression with Gaussian basis and plot the result."""
    w = fit_ridge_regression_gaussian(x, y, mu, s, lam)
    plot_fit_gaussian(x, y, w, mu, s, lam)


def plot_bias_variance_with_val(X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                lam: float):
    """
    Plot training error and validation error vs model complexity (#Gaussians).
    """
    N = len(X_train)
    # number of basis functions between 5%N and 10%N
    gaussian_counts = np.arange(
    max(2, int(0.05 * N)),   # at least 2
    max(3, int(0.1 * N) + 1),
    step=max(1, N // 50)
)


    train_errors = []
    val_errors = []

    for M in gaussian_counts:
        # Compute Gaussian centers using KMeans
        _, mu = kmeans(X_train, M)

        # Width so that Gaussians cover the range
        s = (np.max(X_train) - np.min(X_train)) / (M - 1)

        # Fit model
        w = fit_ridge_regression_gaussian(X_train, y_train, mu, s, lam)

        # Predictions
        y_train_pred = predict_gaussian(X_train, w, mu, s)
        y_val_pred = predict_gaussian(X_val, w, mu, s)

        # Errors
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        train_errors.append(train_mse)
        val_errors.append(val_mse)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(gaussian_counts, train_errors, marker="o", label="Train Error", color="blue")
    plt.plot(gaussian_counts, val_errors, marker="s", label="Validation Error", color="red")
    plt.xlabel("Model Complexity (# of Gaussian Basis Functions)")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Bias-Variance Tradeoff (λ={lam})")
    plt.savefig(f"Assignment-1/Graphs/d1-bias-variance/gaussian-N={N}-{lam}.png")
    plt.legend()
    plt.grid(True)
    plt.show()



train_data = pd.read_csv(r"Assignment-1\team2\Dataset1\train50.csv")
val_data = pd.read_csv(r"Assignment-1\team2\Dataset1\val.csv")


X_train = np.asarray(train_data.drop(columns=["y"]))
y_train = np.asarray(train_data["y"])
X_val = np.asarray(val_data.drop(columns=["y"]))
y_val = np.asarray(val_data["y"])

n = max(2, 0.08*len(X_train))






for l in [0, 0.001, 0.1, 1]:
    plot_bias_variance_with_val(X_train, y_train, X_val, y_val, l)
