import numpy as np
from fir_conv import fir_conv


def log_edge(in_img_array: np.ndarray) -> np.ndarray:
    # Set parameters for the LoG filter
    sigma = 0.6
    log_size = 7
    k = (log_size - 1) / 2

    # Create a grid of (x, y) coordinates centered at zero
    x = np.arange(-k, k, 1)
    x1, x2 = np.meshgrid(x, x)

    # Compute the Laplacian of Gaussian (LoG) mask
    y = (x1 ** 2 + x2 ** 2) / (2 * sigma ** 2)
    h = -1 * (1 - y) * np.exp(-y) / (np.pi * (sigma ** 4))

    # Apply the LoG filter to the input image using convolution
    result_img = fir_conv(in_img_array, h, None, None)[0]

    # Set threshold for zero-crossing detection
    threshold = 0.05
    out_img_array = np.zeros_like(result_img, dtype=int)

    # Loop through each pixel (excluding the border)
    for i in range(1, result_img.shape[0] - 1):
        for j in range(1, result_img.shape[1] - 1):
            center = result_img[i, j]
            if abs(center) > threshold:
                # Symmetrical pairs of pixels around the center pixel
                symmetric_pairs = [
                    (result_img[i, j - 1], result_img[i, j + 1]),  # left - right
                    (result_img[i - 1, j], result_img[i + 1, j]),  # up - down
                    (result_img[i - 1, j - 1], result_img[i + 1, j + 1]),  # diagonally \
                    (result_img[i - 1, j + 1], result_img[i + 1, j - 1])  # diagonally /
                ]

                for a, b in symmetric_pairs:
                    if (a < -threshold and b > threshold) or (a > threshold and b < -threshold):
                        out_img_array[i, j] = 1
                        break

    # Return the binary edge map
    return out_img_array
