import numpy as np


def fir_conv(in_img_array: np.ndarray, h: np.ndarray, in_origin: np.ndarray, mask_origin: np.ndarray) \
        -> [np.ndarray, np.ndarray]:
    img_h, img_w = in_img_array.shape
    mask_h, mask_w = h.shape

    if in_origin is None:
        in_origin = np.array([0, 0], dtype=int)
    if mask_origin is None:
        mask_origin = np.array([mask_h//2, mask_w//2], dtype=int)

    out_origin = in_origin - mask_origin
    out_h = img_h + mask_h - 1
    out_w = img_w + mask_w - 1
    out_img_array = np.zeros((out_h, out_w), dtype=in_img_array.dtype)

    padded_img = np.pad(
        in_img_array,
        ((mask_h - 1, mask_h - 1), (mask_w - 1, mask_w - 1)),
        mode='constant',
        constant_values=0
    )

    h_flipped = np.flip(np.flip(h, axis=0), axis=1)

    for i in range(out_h):
        for j in range(out_w):
            region = padded_img[i:i+mask_h, j:j+mask_w]
            out_img_array[i, j] = np.sum(region * h_flipped)

    # Adjust the output image to the original image size
    out_img_array = out_img_array[mask_h-1:mask_h-1+img_h, mask_w-1:mask_w-1+img_w]

    return out_img_array, out_origin
