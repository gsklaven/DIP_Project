import numpy as np


def image_to_graph(img_array: np.ndarray) -> np.ndarray:
    # Λήψη διαστάσεων εικόνας (ύψος, πλάτος, κανάλια)
    m, n, c = img_array.shape
    num_pixels = m * n  # Συνολικός αριθμός εικονοστοιχείων
    flat_img = img_array.reshape(-1, c)  # Μετατροπή εικόνας σε πίνακα (pixels x κανάλια)

    # Αρχικοποίηση affinity matrix
    affinity_mat = np.zeros((num_pixels, num_pixels), dtype=float)

    # Υπολογισμός ομοιότητας μεταξύ κάθε ζεύγους pixels
    for i in range(num_pixels):
        for j in range(num_pixels):
            diff = flat_img[i] - flat_img[j]  # Διαφορά χαρακτηριστικών μεταξύ δύο pixels
            dist = np.linalg.norm(diff)       # Ευκλείδεια απόσταση
            affinity_mat[i, j] = 1/np.exp(dist)  # Υπολογισμός τιμής ομοιότητας

    return affinity_mat
