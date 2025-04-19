from PIL import Image
import numpy as np
from typing import Dict


def calculate_hist_of_img(img_array: np.ndarray, return_normalized: bool) -> Dict:
    # Αρχικοποίηση λεξικού για το ιστόγραμμα
    hist = {}

    # Εύρεση μοναδικών τιμών
    unique_values = np.unique(img_array)

    # Υπολογισμός του συνολικού αριθμού στοιχείων
    total_samples = img_array.size

    for value in unique_values:
        # Υπολογισμός της συχνότητας εμφάνισης για την τρέχουσα τιμή
        count = 0
        for row in img_array:
            for element in row:
                if element == value:
                    count += 1

        if return_normalized:
            # Κανονικοποίηση συχνότητας εμφάνισης
            hist[value] = count / total_samples
        else:
            # Αποθήκευση απόλυτης συχνότητας
            hist[value] = count

    return hist


def apply_hist_modification_transform(img_array: np.ndarray, modification_transform: Dict) -> np.ndarray:
    # Create a copy of the image array to avoid modifying it in place
    modified_array = img_array.copy()
    for input_value, output_value in modification_transform.items():
        # Use a mask to replace input values with output values
        modified_array[img_array == input_value] = output_value
    return modified_array
