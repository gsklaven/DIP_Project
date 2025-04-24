import numpy as np
from typing import Dict


def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool) -> Dict:
    """
    Calculates the histogram of an image array.

    This function iterates through every unique intensity value in the image and counts its occurrence.
    If normalized histogram is requested, it computes the relative frequency instead of the absolute count.

    Parameters:
        img_array (np.ndarray): The input image array.
        return_normalized (bool): Flag to indicate if the histogram should be normalized.

    Returns:
        Dict: A dictionary with intensity values as keys and counts or normalized frequencies as values.
    """
    hist = {}  # Initialize the histogram dictionary

    # Get all unique intensity values in the image
    unique_values = np.unique(img_array)

    total_samples = img_array.size  # Total number of pixels

    # Iterate over each unique intensity value
    for value in unique_values:
        count = 0  # Initialize counter for current intensity
        # Loop through each row
        for row in img_array:
            # Loop through each element in the row
            for element in row:
                if element == value:
                    count += 1  # Increase count if element equals the current value

        # Store count or normalized frequency in the histogram
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
