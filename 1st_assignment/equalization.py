import numpy as np
import math
from typing import Dict
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform


def greedy(img_array: np.ndarray, hist_ref: Dict) -> Dict:
    n = img_array.size
    lg = np.unique(img_array).size

    gmin = np.min(img_array)
    gmax = np.max(img_array)
    g_levels = np.linspace(gmin, gmax, lg)
    print(f"Lf: {len(np.unique(img_array))}")
    print(f"Οι στάθμες εξόδου g είναι: {g_levels}")

    hist = calculate_hist_of_img(img_array, return_normalized=False)
    print(f"Το ιστογράφημα είναι: {hist}")

    if hist_ref is None:
        pixels_for_g = {g: math.ceil(n / lg) for g in g_levels}
    else:
        pixels_for_g = {g: int(hist_ref.get(g, 0) * n) for g in g_levels}
    print(f"pixels_for_g για κάθε στάθμη: {pixels_for_g}")

    f_keys_sorted = sorted(hist.keys())

    pixel_counter = 0
    g_index = 0
    group_f = []
    modification_transform = {}

    for key in f_keys_sorted:
        print(f"Στάθμη εισόδου: f[{key}], Pixels: {hist[key]}")
        group_f.append(key)
        pixel_counter += hist[key]
        print(f"Συνολικά pixels για g[{g_index}]: {pixel_counter} (τρέχον group: {group_f})")

        if pixel_counter >= pixels_for_g[g_levels[g_index]] or key == f_keys_sorted[-1]:
            print(f"--> Μεταβαίνουμε στη στάθμη εξόδου g[{g_index}] με στάθμες εισόδου {group_f}")
            for f_val in group_f:
                print(f"Αντικαθιστούμε όλα τα pixels με f[{f_val}] στη στάθμη g[{g_levels[g_index]}]")
                modification_transform[f_val] = g_levels[g_index]
            g_index += 1
            pixel_counter = 0
            group_f = []

    return modification_transform


def nongreedy(img_array: np.ndarray, hist_ref: Dict) -> Dict:
    import math
    n = img_array.size
    lg = np.unique(img_array).size

    gmin = np.min(img_array)
    gmax = np.max(img_array)
    g_levels = np.linspace(gmin, gmax, lg)
    print(f"Lf: {len(np.unique(img_array))}")
    print(f"Οι στάθμες εξόδου g είναι: {g_levels}")

    hist = calculate_hist_of_img(img_array, return_normalized=False)
    print(f"Το ιστογράφημα είναι: {hist}")

    if hist_ref is None:
        pixels_for_g = {g: math.ceil(n / lg) for g in g_levels}
    else:
        pixels_for_g = {g: int(hist_ref.get(g, 0) * n) for g in g_levels}
    print(f"pixels_for_g για κάθε στάθμη: {pixels_for_g}")

    f_keys_sorted = sorted(hist.keys())

    g_index = 0
    group_f = []
    pixel_counter = 0
    modification_transform = {}

    for idx, key in enumerate(f_keys_sorted):
        group_f.append(key)
        pixel_counter += hist[key]
        deficiency = pixels_for_g[g_levels[g_index]] - pixel_counter
        print(f"Στάθμη εισόδου: f[{key}], Pixels: {hist[key]}")
        if idx + 1 < len(f_keys_sorted):
            next_count = hist[f_keys_sorted[idx + 1]]
            print(
                f"Σύγκριση: deficiency ({deficiency}) < count(f[next]) / 2 ({next_count / 2}) == "
                f"{deficiency < next_count / 2}"
            )
            if deficiency <= 0 or deficiency < next_count / 2:
                for f_val in group_f:
                    print(f"Αντικαθιστούμε όλα τα pixels με f[{f_val}] στη στάθμη g[{g_levels[g_index]}]")
                    modification_transform[f_val] = g_levels[g_index]
                g_index += 1
                group_f = []
                pixel_counter = 0
        else:
            for f_val in group_f:
                print(f"Αντικαθιστούμε όλα τα pixels με f[{f_val}] στη στάθμη g[{g_levels[g_index]}]")
                modification_transform[f_val] = g_levels[g_index]
            g_index += 1
            group_f = []
            pixel_counter = 0

    return modification_transform


def post_disturbance(img_array: np.ndarray, hist_ref: Dict) -> np.ndarray:
    unique_values = np.unique(img_array)

    d = unique_values[1] - unique_values[0]
    noise = np.random.uniform(low=-d/2, high=d/2, size=img_array.shape)
    disturbed_image = img_array + noise

    min_val = unique_values[0] - d / 2
    max_val = unique_values[-1] + d / 2
    disturbed_image = np.clip(disturbed_image, min_val, max_val)

    disturbed_image = np.round(disturbed_image, 3)
    modification_transform = greedy(disturbed_image, hist_ref)
    g_image = apply_hist_modification_transform(disturbed_image, modification_transform)

    return g_image
