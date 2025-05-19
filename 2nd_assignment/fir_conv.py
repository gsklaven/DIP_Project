import numpy as np

def fir_conv(in_img_array: np.ndarray, h: np.ndarray, in_origin: np.ndarray, mask_origin: np.ndarray) \
        -> [np.ndarray, np.ndarray]:
    # Get dimensions of input image and mask
    img_h, img_w = in_img_array.shape
    mask_h, mask_w = h.shape

    # Set default origins if not provided
    if in_origin is None:
        in_origin = np.array([0, 0], dtype=int)
    if mask_origin is None:
        mask_origin = np.array([mask_h//2, mask_w//2], dtype=int)

    # Calculate output origin and output image size
    out_origin = in_origin - mask_origin
    out_h = img_h + mask_h - 1
    out_w = img_w + mask_w - 1
    out_img_array = np.zeros((out_h, out_w), dtype=in_img_array.dtype)

    # Pad the input image with zeros to handle borders
    padded_img = np.pad(
        in_img_array,
        ((mask_h - 1, mask_h - 1), (mask_w - 1, mask_w - 1)),
        mode='constant',
        constant_values=0
    )

    # Flip the mask for convolution
    h_flipped = np.flip(h, axis=(0, 1))

    # Perform the convolution operation
    for i in range(out_h):
        for j in range(out_w):
            # Extract the region of the padded image
            region = padded_img[i:i+mask_h, j:j+mask_w]
            # Compute the sum of element-wise multiplication
            out_img_array[i, j] = np.sum(region * h_flipped)

    return out_img_array, out_origin