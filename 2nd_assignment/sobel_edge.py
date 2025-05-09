import numpy as np
from fir_conv import fir_conv


def sobel_edge(in_img_array: np.ndarray, thres: float) -> np.ndarray:
    gx1_mask = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=float)

    gx2_mask = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]], dtype=float)

    gx1_img = fir_conv(in_img_array, gx1_mask, None, None)[0]
    gx2_img = fir_conv(in_img_array, gx2_mask, None, None)[0]

    g = np.sqrt(gx1_img ** 2 + gx2_img ** 2)
    out_img_array = (g > thres).astype(int)

    return out_img_array
