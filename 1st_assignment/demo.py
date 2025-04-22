from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform
from hist_modif import perform_hist_modification, perform_hist_eq, perform_hist_matching
import os

output_dir = "./output_plots"
os.makedirs(output_dir, exist_ok=True)

filename = "../images/input_img.jpg"
img = Image.open(fp=filename)
bw_img = img.convert("L")
img_array = np.array(bw_img).astype(float) / 255.0

filename2 = "../images/ref_img.jpg"
img2 = Image.open(fp=filename2)
bw_img2 = img2.convert("L")
img_array2 = np.array(bw_img2).astype(float) / 255.0

# Check calculate_hist_of_img
histogram_normalized = calculate_hist_of_img(img_array, return_normalized=True)
histogram = calculate_hist_of_img(img_array, return_normalized=False)

histogram2_normalized = calculate_hist_of_img(img_array2, return_normalized=True)
histogram2 = calculate_hist_of_img(img_array2, return_normalized=False)

print(f"Ιστόγραμμα κανονικοποιημένο: {histogram_normalized}")
print(f"Ιστόγραμμα: {histogram}")

print(f"Ιστόγραμμα κανονικοποιημένο εικόνας αναφοράς: {histogram2_normalized}")
print(f"Ιστόγραμμα εικόνας αναφοράς: {histogram2}")

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.bar(range(len(histogram_normalized)), histogram_normalized.values(), color="blue", alpha=0.7)
plt.title("Κανονικοποιημένο ιστόγραμμα")
plt.xlabel("Επίπεδο έντασης")
plt.ylabel("Συχνότητα")

plt.subplot(1, 2, 2)
plt.imshow(img_array, cmap="gray")
plt.title("Αρχική Εικόνα")
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "1_original_image_and_histogram.png"))
plt.show()

# Check apply_hist_modification_transform
histogram_transform = apply_hist_modification_transform(img_array, histogram)
print(f"Ιστόγραμμα μετασχηματισμού: {histogram_transform}")

# Check perform_hist_modification
greedy = perform_hist_modification(img_array, histogram2_normalized, "greedy")
nongreedy = perform_hist_modification(img_array, histogram2_normalized, "non-greedy")
post_disturbance = perform_hist_modification(img_array, histogram2_normalized, "post-disturbance")

histogram_greedy = calculate_hist_of_img(greedy, return_normalized=True)
histogram_nongreedy = calculate_hist_of_img(nongreedy, return_normalized=True)
histogram_post_disturbance = calculate_hist_of_img(post_disturbance, return_normalized=True)

plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)
plt.imshow(img_array, cmap="gray")
plt.title("Αρχική Εικόνα")
plt.colorbar()

plt.subplot(2, 4, 2)
plt.imshow(greedy, cmap="gray")
plt.title("Προσέγγιση (Greedy)")
plt.colorbar()

plt.subplot(2, 4, 3)
plt.imshow(nongreedy, cmap="gray")
plt.title("Προσέγγιση (Non-greedy)")
plt.colorbar()

plt.subplot(2, 4, 4)
plt.imshow(post_disturbance, cmap="gray")
plt.title("Προσέγγιση (Post-disturbance)")
plt.colorbar()

plt.subplot(2, 4, 5)
plt.bar(range(len(histogram)), histogram.values(), color="blue", alpha=0.7)
plt.title("Ιστόγραμμα Αρχικής Εικόνας")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.subplot(2, 4, 6)
plt.bar(range(len(histogram_greedy)), histogram_greedy.values(), color="green", alpha=0.7)
plt.title("Προσέγγιση ιστογράμματος Greedy")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.subplot(2, 4, 7)
plt.bar(range(len(histogram_nongreedy)), histogram_nongreedy.values(), color="red", alpha=0.7)
plt.title("Προσέγγιση ιστογράμματος Non-greedy")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.subplot(2, 4, 8)
plt.bar(range(len(histogram_post_disturbance)), histogram_post_disturbance.values(), color="purple", alpha=0.7)
plt.title("Προσέγγιση ιστογράμματος Post-Disturbance")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "2_histogram_modification_comparison.png"))
plt.show()

# Check perform_hist_eq
equalized_img_greedy = perform_hist_eq(img_array, "greedy")
equalized_img_nongreedy = perform_hist_eq(img_array, "non-greedy")
equalized_img_post_disturbance = perform_hist_eq(img_array, "post-disturbance")

histogram_equalized_greedy = calculate_hist_of_img(equalized_img_greedy, return_normalized=True)
histogram_equalized_nongreedy = calculate_hist_of_img(equalized_img_nongreedy, return_normalized=True)
histogram_equalized_post_disturbance = calculate_hist_of_img(equalized_img_post_disturbance, return_normalized=True)

plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)
plt.imshow(img_array, cmap="gray")
plt.title("Αρχική Εικόνα")
plt.colorbar()

plt.subplot(2, 4, 2)
plt.imshow(equalized_img_greedy, cmap="gray")
plt.title("Εξισορρόπηση (Greedy)")
plt.colorbar()

plt.subplot(2, 4, 3)
plt.imshow(equalized_img_nongreedy, cmap="gray")
plt.title("Εξισορρόπηση (Non-greedy)")
plt.colorbar()

plt.subplot(2, 4, 4)
plt.imshow(equalized_img_post_disturbance, cmap="gray")
plt.title("Εξισορρόπηση (Post-disturbance)")
plt.colorbar()

plt.subplot(2, 4, 5)
plt.bar(range(len(histogram)), histogram.values(), color="blue", alpha=0.7)
plt.title("Ιστόγραμμα Αρχικής Εικόνας")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.subplot(2, 4, 6)
plt.bar(range(len(histogram_equalized_greedy)), histogram_equalized_greedy.values(), color="green", alpha=0.7)
plt.title("Εξισορροπημένο Ιστόγραμμα Greedy")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.subplot(2, 4, 7)
plt.bar(range(len(histogram_equalized_nongreedy)), histogram_equalized_nongreedy.values(), color="red", alpha=0.7)
plt.title("Εξισορροπημένο Ιστόγραμμα Non-greedy")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.subplot(2, 4, 8)
plt.bar(range(len(histogram_equalized_post_disturbance)), histogram_equalized_post_disturbance.values(), color="purple",
        alpha=0.7)
plt.title("Εξισορροπημένο Ιστόγραμμα Post-Disturbance")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "3_histogram_equalization_comparison.png"))
plt.show()

# Check perform_hist_matching
matched_img_greedy = perform_hist_matching(img_array, img_array2, "greedy")
matched_img_nongreedy = perform_hist_matching(img_array, img_array2, "non-greedy")
matched_img_post_disturbance = perform_hist_matching(img_array, img_array2, "post-disturbance")

plt.figure(figsize=(16, 12))

# Αρχική εικόνα
plt.subplot(3, 4, 1)
plt.imshow(img_array, cmap="gray")
plt.title("Αρχική Εικόνα")
plt.colorbar()

# Εικόνα αναφοράς
plt.subplot(3, 4, 2)
plt.imshow(img_array2, cmap="gray")
plt.title("Εικόνα Αναφοράς")
plt.colorbar()

# Ιστόγραμμα αρχικής εικόνας
plt.subplot(3, 4, 8)
plt.bar(range(len(histogram)), histogram.values(), color="blue", alpha=0.7)
plt.title("Ιστόγραμμα Αρχικής Εικόνας")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

# Ιστόγραμμα εικόνας αναφοράς
plt.subplot(3, 4, 9)
plt.bar(range(len(histogram2)), histogram2.values(), color="orange", alpha=0.7)
plt.title("Ιστόγραμμα Εικόνας Αναφοράς")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

# Εικόνα μετά το matching (Greedy)
plt.subplot(3, 4, 5)
plt.imshow(matched_img_greedy, cmap="gray")
plt.title("Προσέγγιση (Greedy)")
plt.colorbar()

# Ιστόγραμμα μετά το matching (Greedy)
plt.subplot(3, 4, 10)
plt.bar(range(len(calculate_hist_of_img(matched_img_greedy, return_normalized=True))),
        calculate_hist_of_img(matched_img_greedy, return_normalized=True).values(),
        color="green", alpha=0.7)
plt.title("Προσέγγιση ιστογράμματος  Greedy")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

# Εικόνα μετά το matching (Non-greedy)
plt.subplot(3, 4, 6)
plt.imshow(matched_img_nongreedy, cmap="gray")
plt.title("Προσέγγιση (Non-greedy)")
plt.colorbar()

# Ιστόγραμμα μετά το matching (Non-greedy)
plt.subplot(3, 4, 11)
plt.bar(range(len(calculate_hist_of_img(matched_img_nongreedy, return_normalized=True))),
        calculate_hist_of_img(matched_img_nongreedy, return_normalized=True).values(),
        color="red", alpha=0.7)
plt.title("Προσέγγιση ιστογράμματος  Non-greedy")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

# Εικόνα μετά το matching (Post-disturbance)
plt.subplot(3, 4, 7)
plt.imshow(matched_img_post_disturbance, cmap="gray")
plt.title("Προσέγγιση (Post-disturbance)")
plt.colorbar()

# Ιστόγραμμα μετά το matching (Post-disturbance)
plt.subplot(3, 4, 12)
plt.bar(range(len(calculate_hist_of_img(matched_img_post_disturbance, return_normalized=True))),
        calculate_hist_of_img(matched_img_post_disturbance, return_normalized=True).values(),
        color="purple", alpha=0.7)
plt.title("Προσέγγιση ιστογράμματος Post-Disturbance")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "4_histogram_matching_comparison.png"))
plt.show()
