import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs


def spectral_clustering(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    w = affinity_mat

    d = np.diag(np.sum(w, axis=1))
    la = d - w
    eigen_values, eigen_vectors = eigs(la, k=k, which='SM')
    u = eigen_vectors.real

    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(u)
    labels = kmeans.labels_

    cluster_idx = labels.astype(float)

    return cluster_idx
