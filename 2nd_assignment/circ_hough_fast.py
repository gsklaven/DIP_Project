import numpy as np
from scipy.ndimage import gaussian_filter


def circ_hough(in_img_array: np.ndarray, R_max: float, dim: np.ndarray, V_min: int) -> [np.ndarray, np.ndarray]:
    n1, n2 = in_img_array.shape
    sigma: float = 1.0

    k = np.linspace(0, n1 - 1, dim[0])
    l = np.linspace(0, n2 - 1, dim[1])
    m = np.linspace(2, R_max, dim[2])

    step_k = (n1 - 1) / (dim[0] - 1)
    step_l = (n2 - 1) / (dim[1] - 1)
    step_m = R_max / (dim[2] - 1)

    accumulator = np.zeros((len(k), len(l), len(m)), dtype=int)
    thetas = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    edges = np.argwhere(in_img_array == 1)

    # Υπολογισμός του accumulator
    for xi, yi in edges:
        for r in range(len(m)):
            r_values = m[r]
            a = xi - r_values * np.cos(thetas)
            b = yi - r_values * np.sin(thetas)

            a_index = np.round(a / step_k).astype(int)
            b_index = np.round(b / step_l).astype(int)
            r_index = np.round(r_values / step_m).astype(int)

            for ai, bi in zip(a_index, b_index):
                if 0 <= ai < len(k) and 0 <= bi < len(l):
                    accumulator[ai, bi, r_index] += 1

    # Εφαρμογή του Gaussian φίλτρου στον accumulator
    accumulator = gaussian_filter(accumulator, sigma=sigma)

    centers = []
    radii = []
    print("Max value after Gaussian filtering:", max(accumulator.flatten()))

    window = 6
    half = window // 2

    for a in range(half, len(k) - half):
        for b in range(half, len(l) - half):
            for r in range(half, len(m) - half):
                val = accumulator[a, b, r]
                if val <= V_min:
                    continue
                # Εξαγωγή των γειτονικών τιμών
                neighbors = accumulator[a - half:a + half + 1, b - half:b + half + 1, r - half:r + half + 1]
                max_val = np.max(neighbors)
                if val == max_val and np.count_nonzero(neighbors == val) == 1:
                    centers.append((a * step_k, b * step_l))
                    radii.append(r * step_m)

    return centers, radii
