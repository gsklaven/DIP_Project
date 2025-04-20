import numpy as np
from typing import Dict


def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool) -> Dict:
    hist = {}

    unique_values = np.unique(img_array)

    total_samples = img_array.size

    for value in unique_values:
        count = 0
        for row in img_array:
            for element in row:
                if element == value:
                    count += 1

        if return_normalized:
            hist[value] = count / total_samples
        else:
            hist[value] = count

    return hist


def apply_hist_modification_transform(img_array: np.ndarray, modification_transform: Dict) -> np.ndarray:
    modified_array = img_array.copy()
    for input_value, output_value in modification_transform.items():
        modified_array[img_array == input_value] = output_value
    return modified_array
