import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

# -----------------------------
# Polynomial feature expansion
# -----------------------------
def poly_design_matrix(X: np.ndarray, degree: int, include_bias: bool = True) -> np.ndarray:
    """
    Build polynomial design matrix up to given degree.
    Matches sklearn's PolynomialFeatures ordering.
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    cols = [np.ones((n_samples, 1))] if include_bias else []
    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(n_features), d):
            cols.append(np.prod(X[:, comb], axis=1, keepdims=True))
    return np.hstack(cols) if cols else np.empty((n_samples, 0))

# -----------------------------
# Ridge closed form (multi-output)
# -----------------------------
def fit_poly_ridge_numpy(X: np.ndarray, Y: np.ndarray, degree: int, lam: float):
    """
    Closed-form ridge solution (intercept NOT regularized).
    Returns dict { 'W': (n_poly_features, n_outputs), 'degree': int }
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]

    Phi = poly_design_matrix(X, degree=degree, include_bias=True)
    nphi = Phi.shape[1]

    # Regularization (don't penalize bias)
    R = np.eye(nphi)
    R[0, 0] = 0.0

    A = Phi.T @ Phi + lam * R
    B = Phi.T @ Y

    
    W = np.linalg.pinv(A) @ B

    return {"W": W, "degree": degree}

def predict_poly_ridge_numpy(model, X_new: np.ndarray) -> np.ndarray:
    Phi_new = poly_design_matrix(np.asarray(X_new), degree=model["degree"], include_bias=True)
    return Phi_new @ model["W"]

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))

# -----------------------------
# Load CSVs (train/val/test)
# -----------------------------
train_df = pd.read_csv(r"Assignment-1\team2\Dataset3\dataset3_train.csv")
val_df   = pd.read_csv(r"Assignment-1\team2\Dataset3\dataset3_val.csv")
test_df  = pd.read_csv(r"Assignment-1\team2\Dataset3\dataset3_test.csv")

# --- Set how many outputs live at the END of each CSV ---
n_outputs = 3

# --- Your search grids ---
degrees = [2, 3, 4]
lambdas = [1e-6, 1e-4, 1e-1]

# Split into X (features) and Y (targets)
Xtr, Ytr = train_df.iloc[:, :-n_outputs].values, train_df.iloc[:, -n_outputs:].values
Xva, Yva = val_df.iloc[:, :-n_outputs].values,   val_df.iloc[:, -n_outputs:].values
Xte, Yte = test_df.iloc[:, :-n_outputs].values,  test_df.iloc[:, -n_outputs:].values

model = fit_poly_ridge_numpy(Xtr, Ytr, 2, 0.1)

Yva_hat = predict_poly_ridge_numpy(model, Xva)
Yte_hat = predict_poly_ridge_numpy(model, Xte)


def scatter_true_vs_pred_val_test(Y_val, Y_val_hat, Y_test, Y_test_hat, out_names=None):
    """
    Scatter plots for val and test data in different colours, per output dimension.
    """
    n_out = Y_val.shape[1] if Y_val.ndim > 1 else 1
    if Y_val.ndim == 1:
        Y_val = Y_val[:, None]
        Y_val_hat = Y_val_hat[:, None]
        Y_test = Y_test[:, None]
        Y_test_hat = Y_test_hat[:, None]

    for j in range(n_out):
        lo = float(min(Y_val[:, j].min(), Y_val_hat[:, j].min(),
                       Y_test[:, j].min(), Y_test_hat[:, j].min()))
        hi = float(max(Y_val[:, j].max(), Y_val_hat[:, j].max(),
                       Y_test[:, j].max(), Y_test_hat[:, j].max()))

        plt.figure()
        # Validation points (blue)
        plt.scatter(Y_val[:, j], Y_val_hat[:, j], s=12, color="blue", label="Validation")
        # Test points (green)
        plt.scatter(Y_test[:, j], Y_test_hat[:, j], s=12, color="cyan", label="Test")
        # 45° line (red)
        plt.plot([lo, hi], [lo, hi], color="red", label="Ideal")

        name = out_names[j] if out_names and j < len(out_names) else f"Output {j}"
        plt.xlabel(r"true $t_n$")
        plt.ylabel(r"predicted $y(x_n,\mathbf{w})$")
        plt.title(f"Validation vs Test — {name} (deg=2, λ=0.1)")
        plt.legend()
        plt.savefig(f"{name}.jpg")
        plt.tight_layout()
        plt.show()



scatter_true_vs_pred_val_test(Yva, Yva_hat, Yte, Yte_hat, out_names=['y1', 'y2', 'y3'])
