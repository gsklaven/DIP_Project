from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from hist_utils import calculate_hist_of_img
from equalization import greedy
from equalization import nongreedy
from equalization import post_disturbance
from hist_modif import perform_hist_modification


filename = "../images/input_img.jpg"
img = Image.open(fp=filename)
bw_img = img.convert("L")
img_array = np.array(bw_img).astype(float) / 255.0

best = greedy(img_array, 75)
cist = nongreedy(img_array, 75)
dist = post_disturbance(img_array, 75)

hist = calculate_hist_of_img(img_array, return_normalized=False)
best_hist = calculate_hist_of_img(best, return_normalized=False)
cist_hist = calculate_hist_of_img(cist, return_normalized=False)
dist_hist = calculate_hist_of_img(dist, return_normalized=False)

# Δημιουργία γραφήματος
plt.figure(figsize=(12, 8))

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
plt.title("Εξισορρόπηση (Non-greedy)")
plt.colorbar()

plt.subplot(2, 4, 4)
plt.imshow(dist, cmap="gray")
plt.title("Εξισορρόπηση (Post-Disturbance)")
plt.colorbar()

# Ιστογράμματα (κάτω σειρά)
plt.subplot(2, 4, 5)
plt.bar(range(len(hist)), hist.values(), color="blue", alpha=0.7)
plt.title("Ιστόγραμμα Αρχικής Εικόνας")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.subplot(2, 4, 6)
plt.bar(range(len(best_hist)), best_hist.values(), color="green", alpha=0.7)
plt.title("Ιστόγραμμα Greedy")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.subplot(2, 4, 7)
plt.bar(range(len(cist_hist)), cist_hist.values(), color="red", alpha=0.7)
plt.title("Ιστόγραμμα Non-greedy")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.subplot(2, 4, 8)
plt.bar(range(len(dist_hist)), dist_hist.values(), color="purple", alpha=0.7)
plt.title("Ιστόγραμμα Post-Disturbance")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

# Εμφάνιση του γραφήματος
plt.tight_layout()
plt.show()

