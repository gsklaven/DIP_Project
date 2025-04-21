from typing import Dict
import numpy as np
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform
from equalization import greedy, nongreedy, post_disturbance


def perform_hist_modification(img_array: np.ndarray, hist_ref: Dict, mode: str) -> np.ndarray:
    total_pixels = img_array.size  # Ο συνολικός αριθμός των pixels
    pixels_for_g = {}  # Λεξικό που θα περιέχει τον αριθμό των pixels για κάθε στάθμη εξόδου

    # Υπολογίζουμε το πλήθος των pixels που πρέπει να αναθέσουμε σε κάθε στάθμη εξόδου g
    for level, freq in hist_ref.items():
        pixels_for_g[level] = int(total_pixels * freq)  # Σχετική συχνότητα * συνολικός αριθμός pixels

    print(f"Αριθμός pixels για κάθε στάθμη εξόδου: {pixels_for_g}")

    modified_img = apply_hist_modification_transform(img_array, hist_ref)

    # Επιλογή αλγορίθμου βάσει mode
    if mode == "greedy":
        greedy.pixels_for_g = pixels_for_g  # Δυναμικός καθορισμός του pixels_for_g
        modified_img = greedy(modified_img, len(hist_ref))
    elif mode == "non-greedy":
        nongreedy.pixels_for_g = pixels_for_g  # Δυναμικός καθορισμός του pixels_for_g
        modified_img = nongreedy(modified_img, len(hist_ref))
    elif mode == "post-disturbance":
        post_disturbance.pixels_for_g = pixels_for_g  # Δυναμικός καθορισμός του pixels_for_g
        modified_img = post_disturbance(modified_img, len(hist_ref))
    else:
        raise ValueError("Μη έγκυρη τιμή για το mode.")

    return modified_img
