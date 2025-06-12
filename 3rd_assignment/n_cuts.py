import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs


def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    w = affinity_mat
    d = np.diag(np.sum(w, axis=1))
    la = d - w
    eigen_values, eigen_vectors = eigs(la, k=k, M=d, which='SM')
    u = eigen_vectors.real

    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(u)
    labels = kmeans.labels_

    cluster_idx = labels.astype(float)

    return cluster_idx


def calculate_n_cut_value(affinity_mat: np.ndarray, cluster_idx: np.ndarray) -> float:
    w = affinity_mat
    a = np.where(cluster_idx == 0)[0]
    b = np.where(cluster_idx == 1)[0]
    assoc_a_v = np.sum(w[a, :])
    assoc_a_a = np.sum(w[a[:, None], a])
    assoc_b_v = np.sum(w[b, :])
    assoc_b_b = np.sum(w[b[:, None], b])

    nassoc = assoc_a_a / assoc_a_v + assoc_b_b / assoc_b_v
    n_cut_value = 2 - nassoc

    return n_cut_value


def n_cuts_recursive(affinity_mat: np.ndarray, T1: int, T2: float) -> np.ndarray:
    cluster_idx = n_cuts(affinity_mat, 2)
    a = np.where(cluster_idx == 0)[0]
    b = np.where(cluster_idx == 1)[0]
    n_cut_value = calculate_n_cut_value(affinity_mat, cluster_idx)

    if len(a) < T1 or len(b) < T1 or n_cut_value > T2:
        return cluster_idx

    cluster_idx_a = n_cuts_recursive(affinity_mat[np.ix_(a, a)], T1, T2)
    cluster_idx_b = n_cuts_recursive(affinity_mat[np.ix_(b, b)], T1, T2)

    cluster_idx[a] = cluster_idx_a
    cluster_idx[b] = cluster_idx_b + max(cluster_idx_a) + 1
    return cluster_idx
