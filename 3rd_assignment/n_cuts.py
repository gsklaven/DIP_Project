import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs


# Υλοποίηση του normalized cuts για k clusters
def n_cuts(affinity_mat: np.ndarray, k: int) -> np.ndarray:
    w = affinity_mat  # Πίνακας ομοιότητας
    d = np.diag(np.sum(w, axis=1))  # Διαγώνιος πίνακας βαθμών κόμβων
    la = d - w  # Λαπλασιανή πίνακας γράφου

    eigen_values, eigen_vectors = eigs(la, k=k, M=d, which='SM')
    u = eigen_vectors.real  # Πραγματικό μέρος ιδιοδιανυσμάτων

    # Εφαρμογή k-means clustering στα ιδιοδιανύσματα
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(u)
    labels = kmeans.labels_

    cluster_idx = labels.astype(float)  # Επιστροφή labels ως float

    return cluster_idx


# Υπολογισμός της τιμής normalized cut για δύο clusters
def calculate_n_cut_value(affinity_mat: np.ndarray, cluster_idx: np.ndarray) -> float:
    w = affinity_mat
    a = np.where(cluster_idx == 0)[0]  # Δείκτες για το πρώτο cluster
    b = np.where(cluster_idx == 1)[0]  # Δείκτες για το δεύτερο cluster
    assoc_a_v = np.sum(w[a, :])  # Συνολική συνάφεια του cluster a με όλο το γράφο
    assoc_a_a = np.sum(w[a[:, None], a])  # Συνάφεια εντός του cluster a
    assoc_b_v = np.sum(w[b, :])  # Συνολική συνάφεια του cluster b με όλο το γράφο
    assoc_b_b = np.sum(w[b[:, None], b])  # Συνάφεια εντός του cluster b

    nassoc = assoc_a_a / assoc_a_v + assoc_b_b / assoc_b_v  # Κανονικοποιημένη συνάφεια
    n_cut_value = 2 - nassoc  # Τιμή normalized cut

    return n_cut_value


# Αναδρομική υλοποίηση normalized cuts με όρια μεγέθους και τιμής n-cut
def n_cuts_recursive(affinity_mat: np.ndarray, T1: int, T2: float) -> np.ndarray:
    cluster_idx = n_cuts(affinity_mat, 2)  # Διαχωρισμός σε 2 clusters
    a = np.where(cluster_idx == 0)[0]  # Δείκτες για το πρώτο cluster
    b = np.where(cluster_idx == 1)[0]  # Δείκτες για το δεύτερο cluster
    n_cut_value = calculate_n_cut_value(affinity_mat, cluster_idx)  # Υπολογισμός n-cut

    # Έλεγχος αν πρέπει να σταματήσει η αναδρομή
    if len(a) < T1 or len(b) < T1 or n_cut_value > T2:
        return cluster_idx  # Επιστροφή labels αν ικανοποιείται κάποιο κριτήριο

    # Αναδρομική κλήση για κάθε υπο-cluster
    cluster_idx_a = n_cuts_recursive(affinity_mat[np.ix_(a, a)], T1, T2)
    cluster_idx_b = n_cuts_recursive(affinity_mat[np.ix_(b, b)], T1, T2)

    # Ενημέρωση των labels για το συνολικό cluster
    cluster_idx[a] = cluster_idx_a
    cluster_idx[b] = cluster_idx_b + max(cluster_idx_a) + 1
    return cluster_idx
