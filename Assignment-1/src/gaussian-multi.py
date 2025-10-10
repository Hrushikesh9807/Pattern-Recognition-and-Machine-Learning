#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# K-means (+ k-means++ init)
# =============================
def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initialization for centers."""
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)

    # First center: uniform at random
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]

    # Remaining centers: D^2 sampling
    for i in range(1, k):
        d2_to_existing = np.sum((X[:, None, :] - centers[None, :i, :])**2, axis=2)  # (n, i)
        d2_min = d2_to_existing.min(axis=1)  # (n,)
        probs = d2_min / d2_min.sum()
        next_idx = rng.choice(n, p=probs)
        centers[i] = X[next_idx]
    return centers

def kmeans(X: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-4, random_state: int = 42) -> np.ndarray:
    """
    Simple k-means with k-means++ init. Returns centers with shape (k, n_features).
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    centers = _kmeans_pp_init(X, k, rng)

    for _ in range(max_iter):
        # Assign step
        d2 = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)  # (n_samples, k)
        labels = np.argmin(d2, axis=1)

        # Update step
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = X[mask].mean(axis=0)
            else:
                # Reinitialize empty cluster to a random point
                new_centers[j] = X[rng.integers(0, X.shape[0])]

        # Convergence check
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break
    return centers

# =============================
# RBF design matrix Φ
# =============================
def rbf_design_matrix(X: np.ndarray, centers: np.ndarray, s: float, include_bias: bool = True) -> np.ndarray:
    """
    Gaussian basis: φ_j(x) = exp( -||x - μ_j||^2 / (2 s^2) ), j=1..k
    Returns Φ with shape (n_samples, k [+1 if bias]).
    """
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    d2 = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)  # (n_samples, k)
    Phi = np.exp(-d2 / (2.0 * (s**2)))
    if include_bias:
        Phi = np.concatenate([np.ones((Phi.shape[0], 1)), Phi], axis=1)
    return Phi

# =============================
# Fit / Predict (RBF ridge)
# =============================
def fit_rbf_ridge(X: np.ndarray, Y: np.ndarray, k: int, s: float, lam: float = 0.0, random_state: int = 42):
    """
    Fit multi-output linear regression on RBF features with optional ridge.
    Centers μ_j learned by k-means on X (training only). Bias term is NOT regularized.
    Returns dict with 'centers', 'W', 's', 'k'.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]

    centers = kmeans(X, k=k, random_state=random_state)
    Phi = rbf_design_matrix(X, centers, s, include_bias=True)

    nphi = Phi.shape[1]
    R = np.eye(nphi)
    R[0, 0] = 0.0  # don't regularize bias

    A = Phi.T @ Phi + lam * R
    B = Phi.T @ Y
    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ B

    return {"centers": centers, "W": W, "s": s, "k": k}

def predict_rbf(model, X_new: np.ndarray) -> np.ndarray:
    Phi_new = rbf_design_matrix(np.asarray(X_new, dtype=float),
                                model["centers"], model["s"], include_bias=True)
    return Phi_new @ model["W"]

# =============================
# Data loading helper
# =============================
def load_splits(train_path: str, val_path: str, test_path: str, n_outputs: int, sep: str = ","):
    """
    Reads train, validation, and test CSVs into (X, Y) splits.
    Assumes last n_outputs columns are targets.
    """
    train_df = pd.read_csv(train_path, sep=sep)
    val_df   = pd.read_csv(val_path, sep=sep)
    test_df  = pd.read_csv(test_path, sep=sep)

    Xtr, Ytr = train_df.iloc[:, :-n_outputs].values, train_df.iloc[:, -n_outputs:].values
    Xva, Yva = val_df.iloc[:, :-n_outputs].values,   val_df.iloc[:, -n_outputs:].values
    Xte, Yte = test_df.iloc[:, :-n_outputs].values,  test_df.iloc[:, -n_outputs:].values
    return (Xtr, Ytr), (Xva, Yva), (Xte, Yte)

# =============================
# Metrics
# =============================
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

# =============================
# Plot: Validation + Test together
# =============================
def plot_val_test_scatter(Y_val, Y_val_hat, Y_test, Y_test_hat, erms_val, erms_test, out_names=None, title_prefix="RBF"):
    """
    For each output dimension j:
      - x = true
      - y = predicted
      - blue: validation, green: test
      - red dashed y=x line
      - ERMS shown in legend
    """
    Y_val = np.asarray(Y_val)
    Y_val_hat = np.asarray(Y_val_hat)
    Y_test = np.asarray(Y_test)
    Y_test_hat = np.asarray(Y_test_hat)

    if Y_val.ndim == 1:
        Y_val = Y_val[:, None]
        Y_val_hat = Y_val_hat[:, None]
        Y_test = Y_test[:, None]
        Y_test_hat = Y_test_hat[:, None]

    n_out = Y_val.shape[1]

    for j in range(n_out):
        lo = float(min(Y_val[:, j].min(), Y_val_hat[:, j].min(),
                       Y_test[:, j].min(), Y_test_hat[:, j].min()))
        hi = float(max(Y_val[:, j].max(), Y_val_hat[:, j].max(),
                       Y_test[:, j].max(), Y_test_hat[:, j].max()))
        if lo == hi:
            lo -= 1.0
            hi += 1.0

        plt.figure()
        plt.scatter(Y_val[:, j], Y_val_hat[:, j], s=14, color="blue",  alpha=0.6, label=f"Validation (ERMS={erms_val:.4f})")
        plt.scatter(Y_test[:, j], Y_test_hat[:, j], s=14, color="green", alpha=0.6, label=f"Test (ERMS={erms_test:.4f})")
        plt.plot([lo, hi], [lo, hi], color="red", linestyle="--", label="$y=x$")
        name = out_names[j] if out_names and j < len(out_names) else f"Output {j}"
        plt.xlabel(r"True $y_i$")
        plt.ylabel(r"Predicted $\hat{y}_i$")
        plt.title(f"{title_prefix} — {name}")
        plt.legend()
        plt.tight_layout()
        plt.show()

# =============================
# Main
# =============================
if __name__ == "__main__":
    # --- Paths ---
    train_path = r"Assignment-1\team2\Dataset3\dataset3_train.csv"
    val_path   = r"Assignment-1\team2\Dataset3\dataset3_val.csv"
    test_path  = r"Assignment-1\team2\Dataset3\dataset3_test.csv"

    # --- Targets: last n_outputs columns ---
    n_outputs = 3

    # Load data
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = load_splits(train_path, val_path, test_path, n_outputs)
    print("Shapes:", Xtr.shape, Ytr.shape, "|", Xva.shape, Yva.shape, "|", Xte.shape, Yte.shape)

    # --- Hyperparameter grids ---
    k_grid   = [17, 28, 35]          # number of centers
    s_grid   = [0.1, 0.5, 0.7, 1.0]  # Gaussian spread
    lam_grid = [0.001, 0.1, 1.0]     # ridge lambda

    # Grid search
    results = []
    for k in k_grid:
        for s in s_grid:
            for lam in lam_grid:
                model = fit_rbf_ridge(Xtr, Ytr, k=k, s=s, lam=lam, random_state=42)
                Ytr_hat = predict_rbf(model, Xtr)
                Yva_hat = predict_rbf(model, Xva)
                Yte_hat = predict_rbf(model, Xte)

                tr = rmse(Ytr, Ytr_hat)
                va = rmse(Yva, Yva_hat)
                te = rmse(Yte, Yte_hat)

                results.append({"k": k, "s": s, "lambda": lam, "Train ERMS": tr, "Val ERMS": va, "Test ERMS": te})
                print(f"Done: k={k}, s={s}, λ={lam} | Train {tr:.6f}  Val {va:.6f}  Test {te:.6f}")

    df = pd.DataFrame(results).sort_values(["Val ERMS", "k", "s", "lambda"]).reset_index(drop=True)
    print("\nTop combinations by Val ERMS:\n", df.head(10).to_string(index=False))

    out_csv = "rbf_grid_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved grid results to: {out_csv}")

    # Pick best by validation ERMS
    best = df.iloc[0]
    best_k, best_s, best_lam = int(best["k"]), float(best["s"]), float(best["lambda"])
    print(f"\nBest by Val ERMS -> k={best_k}, s={best_s}, λ={best_lam} "
          f"| Train {best['Train ERMS']:.6f}, Val {best['Val ERMS']:.6f}, Test {best['Test ERMS']:.6f}")

    # Refit best model and make plots
    best_model = fit_rbf_ridge(Xtr, Ytr, k=best_k, s=best_s, lam=best_lam, random_state=42)
    Yva_hat = predict_rbf(best_model, Xva)
    Yte_hat = predict_rbf(best_model, Xte)
    erms_va = rmse(Yva, Yva_hat)
    erms_te = rmse(Yte, Yte_hat)

    plot_val_test_scatter(
        Y_val=Yva, Y_val_hat=Yva_hat,
        Y_test=Yte, Y_test_hat=Yte_hat,
        erms_val=erms_va, erms_test=erms_te,
        out_names=["y1", "y2", "y3"],
        title_prefix=f"Best RBF (k={best_k}, s={best_s}, λ={best_lam})"
    )
