import numpy as np
from fir_conv import fir_conv


def log_edge(in_img_array: np.ndarray) -> np.ndarray:
    log_size = 7.0
    sigma = 1.0
    k = (log_size - 1) / 2
    x = np.arange(-k, k, 1)
    x1, x2 = np.meshgrid(x, x)

    y = (x1 ** 2 + x2 ** 2) / (2 * sigma ** 2)
    h = - 1 * (1 - y) * np.exp(-y) / (np.pi * (sigma ** 4))

    result_img = fir_conv(in_img_array, h, None, None)[0]

    out_img_array = np.zeros_like(result_img, dtype=int)

    for i in range(1, result_img.shape[0] - 1):
        for j in range(1, result_img.shape[1] - 1):
            center = result_img[i, j]
            neighborhood = result_img[i - 1:i + 2, j - 1:j + 2]
            if (center > 0 and np.any(neighborhood < 0)) or (center < 0 and np.any(neighborhood > 0)):
                out_img_array[i, j] = 1

    return out_img_array
