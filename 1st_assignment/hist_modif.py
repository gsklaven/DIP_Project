import numpy as np
from typing import Dict
from PIL import Image
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform
from equalization import greedy, nongreedy, post_disturbance


def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:
    total_pixels = img_array.size
    pixels_for_g = {}

    for level, freq in hist_ref.items():
        pixels_for_g[level] = int(total_pixels * freq)
    print(f"Αριθμός pixels για κάθε στάθμη εξόδου: {pixels_for_g}")

    modified_img = apply_hist_modification_transform(img_array, hist_ref)

    # Επιλογή αλγορίθμου βάσει mode
    if mode == "greedy":
        greedy.pixels_for_g = pixels_for_g
        modified_img = greedy(modified_img, len(hist_ref))
    elif mode == "non-greedy":
        nongreedy.pixels_for_g = pixels_for_g
        modified_img = nongreedy(modified_img, len(hist_ref))
    elif mode == "post-disturbance":
        post_disturbance.pixels_for_g = pixels_for_g
        modified_img = post_disturbance(modified_img, len(hist_ref))
    else:
        raise ValueError("Μη έγκυρη τιμή για το mode.")

    return modified_img


def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    hist = calculate_hist_of_img(img_array, return_normalized=True)
    equalized_img = perform_hist_modification(img_array, hist, mode)

    return equalized_img


def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str) -> np.ndarray:
    hist_ref = calculate_hist_of_img(img_array_ref, return_normalized=True)
    print(f"Ιστόγραμμα αναφοράς: {hist_ref}")

    processed_img = perform_hist_modification(img_array, hist_ref, mode)

    return processed_img
