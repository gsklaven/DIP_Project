import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs


def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    w = affinity_mat

    d = np.diag(np.sum(w, axis=1))
    l = d - w
    eigen_values, eigen_vectors = eigs(l, k=k, M=d, which='SM')
    u = eigen_vectors.real

    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(u)
    labels = kmeans.labels_

    cluster_idx = labels.astype(float)

    return cluster_idx


def calculate_n_cut_value(affinity_mat: np.ndarray, cluster_idx: np.ndarray) -> float:

    return n_cut_value
