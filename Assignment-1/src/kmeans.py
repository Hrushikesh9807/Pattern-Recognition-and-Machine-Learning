import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def initialize_centroids(X, k):
    """Randomly pick k points as initial centroids."""
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    """Assign each point to the nearest centroid."""
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # (n_samples, k)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    """Update centroids as mean of points in each cluster."""
    return np.array([X[labels == j].mean(axis=0) for j in range(k)])

def kmeans(X, k, max_iters=100, tol=1e-4):
    """Run K-Means clustering on X."""
    X = np.asarray(X, dtype=float)

    # ensure shape (n_samples, n_features)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
        centroids = new_centroids
    return labels, centroids

def plot_clusters(X, labels, centroids):
    """Plot clusters (works for 1D or 2D data)."""
    if X.shape[1] == 1:  # 1D data
        plt.scatter(X[:, 0], np.zeros_like(X[:, 0]), c=labels, cmap="viridis")
        plt.scatter(centroids[:, 0], np.zeros_like(centroids[:, 0]), 
                    c="red", marker="x", s=200)
        plt.xlabel("X")
        plt.title("K-Means Clustering (1D)")
    elif X.shape[1] == 2:  # 2D data
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                    c="red", marker="x", s=200)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("K-Means Clustering (2D)")
    else:
        print("Plotting not supported for dimension > 2")
        return
    plt.show()


# data = pd.read_csv(r"team2\Dataset2\train100.csv")
# x = np.asarray(data.drop(columns=['y']))
# y = np.asarray(data['y'])

# n = 3
# labels, centroids = kmeans(x, n)
# print(centroids)
# plot_clusters(x, labels, centroids)