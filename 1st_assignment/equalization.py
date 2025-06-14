import numpy as np
import math
from typing import Dict
from hist_utils import apply_hist_modification_transform, calculate_hist_of_img


def greedy(img_array: np.ndarray, hist: Dict, hist_ref: Dict) -> Dict:
    """
        Applies the greedy histogram modification approach to compute a transformation mapping.

        This function computes output intensity levels based on a greedy method, grouping input intensity
        levels until a threshold (computed from the reference histogram or equal distribution) is met.
        The transformation maps input levels to output levels accordingly.

        Parameters:
            img_array (np.ndarray): The input image as a 2D numpy array.
            hist (Dict): The histogram of the input image (non-normalized).
            hist_ref (Dict): The reference histogram normalized values, or None for histogram equalization.

        Returns:
            Dict: A dictionary that maps each input intensity level to the corresponding output level.
        """
    # Compute total number of pixels and number of unique intensity levels.
    n = img_array.size
    lg = np.unique(img_array).size

    # Determine the minimum and maximum intensity values.
    gmin = np.min(img_array)
    gmax = np.max(img_array)

    # Create evenly spaced output levels.
    g_levels = np.linspace(gmin, gmax, lg)
    print(f"Lf: {len(np.unique(img_array))}")
    print(f"Output levels g: {g_levels}")

    # Determine the number of pixels required for each output level
    # using either the reference histogram or equal distribution.
    if hist_ref is None:
        pixels_for_g = {g: math.ceil(n / lg) for g in g_levels}
    else:
        pixels_for_g = {g: int(hist_ref.get(g, 0) * n) for g in g_levels}
    print(f"Pixels for each level: {pixels_for_g}")

    # Sort histogram keys to process intensity levels in order.
    f_keys_sorted = sorted(hist.keys())

    pixel_counter = 0  # Accumulates pixel counts for the current group.
    g_index = 0  # Index for the output levels.
    group_f = []  # Holds current group of input intensity levels.
    modification_transform = {}  # Mapping of input intensity to output intensity.

    # Process each intensity level sequentially.
    for key in f_keys_sorted:
        print(f"Input level: f[{key}], Pixels: {hist[key]}")
        group_f.append(key)
        pixel_counter += hist[key]
        print(f"Total pixels for g[{g_index}]: {pixel_counter} (current group: {group_f})")

        # If the accumulated pixel count meets the target, or it's the final level,
        # assign the mapping for all intensity levels in the group.
        if pixel_counter >= pixels_for_g[g_levels[g_index]] or key == f_keys_sorted[-1]:
            print(f"--> Transitioning to output level g[{g_index}] with input levels {group_f}")
            for f_val in group_f:
                print(f"Mapping input level f[{f_val}] to output level g[{g_levels[g_index]}]")
                modification_transform[f_val] = g_levels[g_index]
            g_index += 1  # Move to the next output level.
            pixel_counter = 0  # Reset the counter.
            group_f = []  # Reset the group.

    return modification_transform


def nongreedy(img_array: np.ndarray, hist: Dict, hist_ref: Dict) -> Dict:
    """
        Applies the non-greedy histogram modification approach to compute a transformation mapping.

        This function groups intensity levels and compares the deficiency (remaining required pixels for the
        current output level) with half of the next level's pixel count to determine when to shift to the next
        output level.

        Parameters:
            img_array (np.ndarray): The input image as a 2D numpy array.
            hist (Dict): The histogram of the input image (non-normalized).
            hist_ref (Dict): The reference histogram normalized values, or None for histogram equalization.

        Returns:
            Dict: A dictionary mapping each input intensity level to the corresponding output intensity level.
        """
    # Compute total pixel count and number of unique levels.
    n = img_array.size
    lg = np.unique(img_array).size

    # Determine the intensity range and set up equally spaced output levels.
    gmin = np.min(img_array)
    gmax = np.max(img_array)
    g_levels = np.linspace(gmin, gmax, lg)
    print(f"Lf: {len(np.unique(img_array))}")
    print(f"Output levels g: {g_levels}")

    # Calculate target pixels count per output level based on the reference histogram
    # or equal distribution when no reference is provided.
    if hist_ref is None:
        pixels_for_g = {g: math.ceil(n / lg) for g in g_levels}
    else:
        pixels_for_g = {g: int(hist_ref.get(g, 0) * n) for g in g_levels}
    print(f"Pixels for each level: {pixels_for_g}")

    # Sort the input intensity levels.
    f_keys_sorted = sorted(hist.keys())

    g_index = 0  # Output level index.
    group_f = []  # Group of intensity levels to be mapped.
    pixel_counter = 0  # Accumulates pixels in the current group.
    modification_transform = {}  # Final mapping dictionary.

    # Loop through sorted intensity levels.
    for idx, key in enumerate(f_keys_sorted):
        group_f.append(key)
        pixel_counter += hist[key]
        # Calculate deficit: how many more pixels needed for current output level.
        deficiency = pixels_for_g[g_levels[g_index]] - pixel_counter
        print(f"Input level: f[{key}], Pixels: {hist[key]}")
        # Check if the next level's pixel count would overshoot the target.
        if idx + 1 < len(f_keys_sorted):
            next_count = hist[f_keys_sorted[idx + 1]]
            print(
                f"Comparison: deficiency ({deficiency}) < next level count/2 ({next_count / 2}) == "
                f"{deficiency < next_count / 2}"
            )
            # If requirements are met or nearly met, map the group to the current output level.
            if deficiency <= 0 or deficiency < next_count / 2:
                for f_val in group_f:
                    print(f"Mapping input level f[{f_val}] to output level g[{g_levels[g_index]}]")
                    modification_transform[f_val] = g_levels[g_index]
                g_index += 1  # Proceed to the next output level.
                group_f = []  # Reset the group.
                pixel_counter = 0  # Reset the counter.
        else:
            # For the last intensity level, map the remaining group.
            for f_val in group_f:
                print(f"Mapping input level f[{f_val}] to output level g[{g_levels[g_index]}]")
                modification_transform[f_val] = g_levels[g_index]
            g_index += 1
            group_f = []
            pixel_counter = 0

    return modification_transform


def post_disturbance(img_array: np.ndarray, hist: dict, hist_ref: dict) -> np.ndarray:
    """
    Applies histogram modification after introducing uniform random noise to disturb the pixel values.

    This function adds noise over the full bucket width, quantizes the disturbed image, computes its histogram,
    and then applies the greedy approach to compute and apply a transformation mapping. This allows pixels from a densely
    populated input level to spread into neighboring levels.

    Parameters:
        img_array (np.ndarray): Input image array with normalized values in [0, 1].
        hist (dict): Histogram of the original image (counts per intensity level).
        hist_ref (dict): Reference histogram normalized values for matching or None for equalization.

    Returns:
        np.ndarray: Histogram modified image.
    """
    # Get unique intensity levels and compute level spacing (assumes equal spacing).
    orig_levels = np.unique(img_array)
    d = orig_levels[1] - orig_levels[0]

    # Add uniform noise in the full bucket width [-d, d] to disturb the image.
    noise = np.random.uniform(low=-d/2, high=d/2, size=img_array.shape)
    disturbed_image = img_array + noise

    # Clip the disturbed image to ensure pixel values remain within [0, 1].
    disturbed_image = np.clip(disturbed_image, 0, 1)

    # Quantize the disturbed image by rounding each pixel to the nearest original intensity level.
    quantized_image = np.round(disturbed_image / d) * d
    quantized_image = np.clip(quantized_image, 0, 1)

    # Compute the histogram of the quantized (disturbed) image.
    disturbed_hist = calculate_hist_of_img(quantized_image, return_normalized=False)

    # Compute the transformation mapping using the greedy approach and the disturbed histogram.
    modification_transform = greedy(quantized_image, disturbed_hist, hist_ref)

    # Apply the transformation mapping to get the final modified image.
    modified_img = apply_hist_modification_transform(quantized_image, modification_transform)

    return modified_img
