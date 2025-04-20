from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from hist_utils import calculate_hist_of_img
from hist_utils import apply_hist_modification_transform
from equalization import greedy
from equalization import nongreedy
from equalization import post_disturbance
from hist_modif import perform_hist_modification


filename = "../images/posterized_image.jpg"
img = Image.open(fp=filename)
bw_img = img.convert("L")
img_array = np.array(bw_img).astype(float) / 255.0

# img_array = np.array([
#     [0.0, 0.0, 0.2, 0.3],
#     [0.2, 0.3, 0.4, 0.5],
#     [0.5, 0.6, 0.7, 0.7],
#     [0.8, 0.8, 0.9, 1.0],
#     [1.0, 1.0, 1.0, 0.9],
#     [0.9, 0.8, 0.8, 0.7],
#     [0.6, 0.5, 0.5, 0.4],
# ])

hist = calculate_hist_of_img(img_array, return_normalized=True)
print(hist)

best = greedy(img_array, 75)
print(best)

cist = nongreedy(img_array, 75)
print(cist)

dist = post_disturbance(img_array, 75)
print(dist)

plt.figure(figsize=(12, 10))

# Εικόνες (πάνω σειρά)
plt.subplot(2, 4, 1)
plt.imshow(img_array, cmap="gray")
plt.title("Αρχική Εικόνα")
plt.colorbar()

plt.subplot(2, 4, 2)
plt.imshow(best, cmap="gray")
plt.title("Εξισορρόπηση (Greedy)")
plt.colorbar()

plt.subplot(2, 4, 3)
plt.imshow(cist, cmap="gray")
plt.title("Εξισορρόπηση (Nongreedy)")
plt.colorbar()

plt.subplot(2, 4, 4)
plt.imshow(dist, cmap="gray")
plt.title("Εξισορρόπηση (Post-Disturbance)")
plt.colorbar()

# Ιστογράμματα (κάτω σειρά)
plt.subplot(2, 4, 5)
plt.bar(range(len(hist)), hist, color="blue")
plt.title("Ιστόγραμμα Αρχικής Εικόνας")

best_hist = calculate_hist_of_img(best, return_normalized=True)
plt.subplot(2, 4, 6)
plt.bar(range(len(best_hist)), best_hist, color="green")
plt.title("Ιστόγραμμα Greedy")

cist_hist = calculate_hist_of_img(cist, return_normalized=True)
plt.subplot(2, 4, 7)
plt.bar(range(len(cist_hist)), cist_hist, color="red")
plt.title("Ιστόγραμμα Nongreedy")

dist_hist = calculate_hist_of_img(dist, return_normalized=False)
plt.subplot(2, 4, 8)
plt.bar(range(len(dist_hist)), dist_hist, color="purple")
plt.title("Ιστόγραμμα Post-Disturbance")

plt.tight_layout()
plt.show()
