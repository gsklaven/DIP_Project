import numpy as np
from typing import Dict
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform
from equalization import greedy, nongreedy, post_disturbance


def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:

    if mode == "greedy":
        modification_transform = greedy(img_array, hist_ref)
        modified_img = apply_hist_modification_transform(img_array, modification_transform)
    elif mode == "non-greedy":
        modification_transform = nongreedy(img_array, hist_ref)
        modified_img = apply_hist_modification_transform(img_array, modification_transform)
    elif mode == "post-disturbance":
        modified_img = post_disturbance(img_array, hist_ref)
    else:
        raise ValueError("Unknown mode")

    return modified_img


def perform_hist_eq(img_array: np.ndarray, mode: str) -> np.ndarray:
    equalized_img = perform_hist_modification(img_array, None, mode)
    return equalized_img


def perform_hist_matching(img_array: np.ndarray, img_array_ref: np.ndarray, mode: str) -> np.ndarray:
    hist_ref = calculate_hist_of_img(img_array_ref, return_normalized=True)
    print(f"Ιστόγραμμα αναφοράς: {hist_ref}")

    processed_img = perform_hist_modification(img_array, hist_ref, mode)

    return processed_img
