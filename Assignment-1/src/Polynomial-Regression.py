import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from itertools import combinations_with_replacement
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error



def design_matrix(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Construct polynomial design matrix Φ for multivariate input X.
    Works for numpy arrays (n_samples, n_features).
    """
    X = np.asarray(X)   # ensure numpy
    n_samples, n_features = X.shape
    features = [np.ones(n_samples)]  # bias term

    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(n_features), d):
            feat = np.prod(X[:, comb], axis=1)
            features.append(feat)

    return np.column_stack(features)



def fit_ridge_regression(X: np.ndarray, y: np.ndarray, degree: int, lam: float) -> np.ndarray:
    """Fit polynomial regression with ridge regularization."""
    Phi = design_matrix(X, degree)
    print()
    identity = np.identity(Phi.shape[1])
    return np.linalg.inv(lam * identity + Phi.T @ Phi) @ Phi.T @ y


def predict(X: np.ndarray, w: np.ndarray, degree: int) -> np.ndarray:
    """Predict outputs for given inputs using fitted weights."""
    Phi = design_matrix(X, degree)
    return Phi @ w


def plot_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, degree: int, lam: float) -> None:
    """Plot data and fitted curve (for 1D input)."""
    if X.shape[1] != 1:
        raise ValueError("Plotting is only supported for 1D input data")

    x_range = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
    y_pred = predict(x_range, w, degree)

    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(x_range, y_pred, color="red", label="Model")
    plt.title(f"Polynomial Regression (degree={degree}, λ={lam}, N={len(X)})")
    plt.legend()
    plt.show()


def plot_fit_surface(X: np.ndarray, y: np.ndarray, w: np.ndarray, degree: int, lam: float) -> None:
    """Plot data and fitted surface (for 2D input)."""
    if X.shape[1] != 2:
        raise ValueError("Surface plotting is only supported for 2D input data")

    x1_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 50)
    x2_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 50)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    grid_points = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
    y_pred_grid = predict(grid_points, w, degree).reshape(x1_grid.shape)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y, color="blue", label="Data")
    ax.plot_surface(x1_grid, x2_grid, y_pred_grid, cmap="viridis", alpha=0.7)

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("y")
    ax.set_title(f"Polynomial Ridge Regression (degree={degree}, λ={lam}, N={len(X)})")
    plt.legend()
    plt.show()
    plt.savefig(f"Graphs/dataset-2/poly-{degree}-{lam}-{len(x)}.jpg")


def solve(X: np.ndarray, y: np.ndarray, degree: int, lam: float) -> None:
    """Solve ridge regression and plot the result (1D or 2D)."""
    w = fit_ridge_regression(X, y, degree, lam)
    if X.shape[1] == 1:
        plot_fit(X, y, w, degree, lam)
    elif X.shape[1] == 2:
        plot_fit_surface(X, y, w, degree, lam)
    else:
        print("Plotting not supported for dimension > 2")
    return w



def plot_bias_variance_with_val(X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                degrees: list, lam: float):
    """
    Plot training error and validation error vs model complexity (degree).
    """
    train_errors = []
    val_errors = []

    for d in degrees:
        # Fit model on training data
        w = fit_ridge_regression(X_train, y_train, d, lam)

        # Predictions
        y_train_pred = predict(X_train, w, d)
        y_val_pred = predict(X_val, w, d)

        # Errors
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        train_errors.append(train_mse)
        val_errors.append(val_mse)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, train_errors, marker="o", label="Train Error", color="blue")
    plt.plot(degrees, val_errors, marker="s", label="Validation Error", color="red")
    plt.xlabel("Model Complexity (Polynomial Degree)")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Bias-Variance Tradeoff (λ={lam})")
    plt.savefig(f"Assignment-1/Graphs/d1-bias-variance/N={len(X_train)}-{lam}.png")
    plt.legend()
    plt.grid(True)
    plt.show()


def erms(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the root mean squared error (ERMS)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def print_erms_table(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     degrees: list, lam_values: list):
    """
    Print a table showing ERMS on train, val, and test sets
    for models without and with regularization.
    """
    rows = []

    for lam in lam_values:
        for d in degrees:
            # Fit model
            w = fit_ridge_regression(X_train, y_train, d, lam)

            # Predictions
            y_train_pred = predict(X_train, w, d)
            y_val_pred = predict(X_val, w, d)
            y_test_pred = predict(X_test, w, d)

            # Compute ERMS
            train_erms = erms(y_train, y_train_pred)
            val_erms = erms(y_val, y_val_pred)
            test_erms = erms(y_test, y_test_pred)

            rows.append({
                "Degree": d,
                "Lambda": lam,
                "Train ERMS": round(train_erms, 4),
                "Validation ERMS": round(val_erms, 4),
                "Test ERMS": round(test_erms, 4)
            })

    # Convert to DataFrame for pretty display
    df = pd.DataFrame(rows)
    df.to_csv(r"Assignment-1/results/Table-3.csv")
    print(df.to_string(index=False))
    return df




train_data = pd.read_csv(r"Assignment-1\team2\Dataset3\dataset3_train.csv")
val_data = pd.read_csv(r"Assignment-1\team2\Dataset3\dataset3_val.csv")

X_train = np.asarray(train_data.drop(columns=["y"]))
y_train = np.asarray(train_data["y"])
X_val = np.asarray(val_data.drop(columns=["y"]))
y_val = np.asarray(val_data["y"])

deg = [3, 5, 7, 9]

for l in [0, 0.001, 0.1, 1]:
    plot_bias_variance_with_val(X_train, y_train, X_val, y_val, deg, l)





