import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs


# Υλοποίηση του spectral clustering με χρήση affinity matrix και k clusters
def spectral_clustering(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    w = affinity_mat

    d = np.diag(np.sum(w, axis=1))  # Διαγώνιος πίνακας βαθμών κόμβων
    la = d - w  # Λαπλασιανή πίνακας γράφου

    eigen_values, eigen_vectors = eigs(la, k=k, which='SM')
    u = eigen_vectors.real  # Πραγματικό μέρος ιδιοδιανυσμάτων

    # Εφαρμογή k-means clustering στα ιδιοδιανύσματα
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(u)
    labels = kmeans.labels_

    cluster_idx = labels.astype(float)  # Επιστροφή labels ως float

    return cluster_idx
