import numpy as np
from typing import Dict
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform
from equalization import greedy, nongreedy, post_disturbance


def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:
    """
    Performs histogram modification based on a specified mode.

    This function calculates the image histogram, then applies one of the histogram modification
    methods (greedy, non-greedy, or post-disturbance) using the provided reference histogram
    (or None for equalization).

    Parameters:
        img_array (np.ndarray): The input image array.
        hist_ref (Dict): The reference histogram for histogram matching (or None for equalization).
        mode (str): The modification mode ('greedy', 'non-greedy', or 'post-disturbance').

    Returns:
        np.ndarray: The histogram modified image.
    """
    # Calculate the histogram of the input image (non-normalized)
    hist = calculate_hist_of_img(img_array, return_normalized=False)
    print(f"Histogram: {hist}")

    # Select modification method according to the mode parameter
    if mode == "greedy":
        modification_transform = greedy(img_array, hist, hist_ref)
        modified_img = apply_hist_modification_transform(img_array, modification_transform)
    elif mode == "non-greedy":
        modification_transform = nongreedy(img_array, hist, hist_ref)
        modified_img = apply_hist_modification_transform(img_array, modification_transform)
    elif mode == "post-disturbance":
        modified_img = post_disturbance(img_array, hist, hist_ref)
    else:
        raise ValueError("Unknown mode")

    return modified_img


def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    """
    Performs histogram equalization on the input image.

    Uses histogram modification methods with no reference histogram.

    Parameters:
        img_array (np.ndarray): The input image array.
        mode (str): The modification mode.

    Returns:
        np.ndarray: The equalized image.
    """
    # Call perform_hist_modification with hist_ref set to None for equalization
    equalized_img = perform_hist_modification(img_array, None, mode)
    return equalized_img


def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str) -> np.ndarray:
    """
    Performs histogram matching between an input image and a reference image.

    The function computes the normalized histogram of the reference image and uses it as input
    to the histogram modification method.

    Parameters:
        img_array (np.ndarray): The input image array.
        img_array_ref (np.ndarray): The reference image array.
        mode (str): The modification mode.

    Returns:
        np.ndarray: The histogram matched image.
    """
    # Calculate normalized histogram of the reference image
    hist_ref = calculate_hist_of_img(img_array_ref, return_normalized=True)
    print(f"Reference histogram: {hist_ref}")

    # Process the input image using the computed reference histogram
    processed_img = perform_hist_modification(img_array, hist_ref, mode)
    return processed_img
