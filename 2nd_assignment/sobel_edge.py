import numpy as np
from fir_conv import fir_conv


def sobel_edge(in_img_array: np.ndarray, thres: float) -> np.ndarray:
    # Define Sobel masks for horizontal and vertical edge detection
    gx1_mask = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=float)

    gx2_mask = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]], dtype=float)

    # Apply convolution with the Sobel masks
    gx1_img = fir_conv(in_img_array, gx1_mask, None, None)[0]
    gx2_img = fir_conv(in_img_array, gx2_mask, None, None)[0]

    # Compute the gradient magnitude
    g = np.sqrt(gx1_img ** 2 + gx2_img ** 2)

    # Apply threshold to get binary edge map
    out_img_array = (g > thres).astype(int)

    return out_img_array
