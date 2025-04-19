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


# Δημιουργία παραδείγματος εικόνας
img_array = np.array([[0.1, 0.2, 0.1], [0.3, 0.1, 0.3], [0.2, 0.3, 0.3]], dtype=float)

filename = "images/input_img.jpg"
img = Image.open(fp=filename)
bw_img = img.convert("L")
#img_array = np.array(bw_img).astype(float) / 255.0

hist = calculate_hist_of_img(img_array, return_normalized=True)
print(hist)

modified_image = apply_hist_modification_transform(img_array, hist)
print(modified_image)
