import numpy as np


def image_to_graph(img_array: np.ndarray) -> np.ndarray:
    m, n, c = img_array.shape
    num_pixels = m * n
    flat_img = img_array.reshape(-1, c)

    affinity_mat = np.zeros((num_pixels, num_pixels), dtype=float)

    for i in range(num_pixels):
        for j in range(num_pixels):
            diff = flat_img[i] - flat_img[j]
            dist = np.linalg.norm(diff)
            affinity_mat[i, j] = 1/np.exp(dist)

    return affinity_mat
