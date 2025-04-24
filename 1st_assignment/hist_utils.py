import numpy as np
from typing import Dict


def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool) -> Dict:
    """
    Calculates the histogram of an image array.

    This function iterates through every unique intensity value in the image and counts its occurrence.
    If normalized histogram is requested, it computes the relative frequency instead of the absolute count.

    Parameters:
        img_array (np.ndarray): The input image array.
        return_normalized (bool): Whether to normalize the histogram.

    Returns:
        Dict: A histogram dictionary with intensity values as keys.
    """
    hist = {}

    # Flatten the image so we can work with 1D array
    flat_img = img_array.flatten()

    # Get unique values and their counts
    unique_values = np.unique(flat_img)

    total_samples = flat_img.size

    for value in unique_values:
        # Count occurrences using boolean masking
        count = np.sum(flat_img == value)

        if return_normalized:
            hist[value] = count / total_samples
        else:
            hist[value] = count

    return hist


def apply_hist_modification_transform(img_array: np.ndarray, modification_transform: Dict) -> np.ndarray:
    """
    Applies a given histogram modification transformation on an image.

    This function creates a copy of the input image and replaces each pixel value with its corresponding
    output value defined in the transformation mapping.

    Parameters:
        img_array (np.ndarray): The input image array.
        modification_transform (Dict): A dictionary mapping input intensity values to output intensity values.

    Returns:
        np.ndarray: The modified image after applying the transformation.
    """
    modified_img = img_array.copy()  # Create a copy of the input image
    # Loop through each mapping entry and update the image
    for input_value, output_value in modification_transform.items():
        modified_img[img_array == input_value] = output_value  # Replace pixels matching the input value

    return modified_img
