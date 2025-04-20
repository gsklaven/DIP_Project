from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from hist_utils import calculate_hist_of_img
from hist_utils import apply_hist_modification_transform
from equalization import greedy
from equalization import nongreedy
from equalization import post_disturbance
from hist_modif import perform_hist_modification


filename = "../images/input_img.jpg"
img = Image.open(fp=filename)
bw_img = img.convert("L")
# img_array = np.array(bw_img).astype(float) / 255.0

# img_array = np.array([
#     [0.0, 0.0, 0.0, 0.3],
#     [0.3, 0.3, 0.6, 0.6],
#     [0.8, 0.8, 1.0, 1.0],
#     [1.0, 1.0, 1.0, 1.0],
#     [1.0, 1.0, 1.0, 1.0],
#     [1.0, 1.0, 1.0, 1.0],
#     [0.0, 0.0, 0.0, 0.3],
# ])

# hist = calculate_hist_of_img(img_array, return_normalized=True)
# print(hist)

# test = apply_hist_modification_transform(img_array, hist)
# print(test)
#
# hist = {0.0: 0.25, 0.3: 0.25, 0.6: 0.25, 1.0: 0.25}
#
# gist = perform_hist_modification(img_array, hist, "greedy")
# print(gist)

# best = greedy(img_array, 5)
# print(best)
#
# cist = nongreedy(img_array, 5)
# print(cist)
#
# dist = post_disturbance(img_array, 5)
# print(dist)
#
# # Αρχική εικόνα
# plt.figure(figsize=(10, 8))
# plt.subplot(2, 3, 1)
# plt.imshow(img_array, cmap="gray")
# plt.title("Αρχική Εικόνα")
# plt.colorbar()
#
# # Αρχικό Ιστόγραμμα
# plt.subplot(2, 3, 2)
# plt.bar(range(len(hist)), hist, color="blue")
# plt.title("Ιστόγραμμα Αρχικής Εικόνας")
#
# # Εφαρμογή Greedy
# plt.subplot(2, 3, 3)
# plt.imshow(best, cmap="gray")
# plt.title("Εξισορρόπηση (Greedy)")
# plt.colorbar()
#
# # Εφαρμογή Nongreedy
# plt.subplot(2, 3, 4)
# plt.imshow(cist, cmap="gray")
# plt.title("Εξισορρόπηση (Nongreedy)")
# plt.colorbar()
#
# # Εφαρμογή Post-Disturbance
# plt.subplot(2, 3, 5)
# plt.imshow(dist, cmap="gray")
# plt.title("Εξισορρόπηση (Post-Disturbance)")
# plt.colorbar()
#
# plt.tight_layout()
# plt.show()

img_array = np.array([
    [0.0, 0.0, 0.2, 0.3],
    [0.2, 0.3, 0.4, 0.5],
    [0.5, 0.6, 0.7, 0.7],
    [0.8, 0.8, 0.9, 1.0],
    [1.0, 1.0, 1.0, 0.9],
    [0.9, 0.8, 0.8, 0.7],
    [0.6, 0.5, 0.5, 0.4],
])

hist_ref = {
    0.1: 0.1,  # 10% των pixels
    0.3: 0.3,  # 30% των pixels
    0.5: 0.2,  # 20% των pixels
    0.7: 0.2,  # 20% των pixels
    1.0: 0.2   # 20% των pixels
}
mode = "greedy"

result = perform_hist_modification(img_array, hist_ref, mode)

# Εκτύπωση αποτελέσματος
print("Τελική Εικόνα:")
print(result)
