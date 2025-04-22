from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from hist_utils import calculate_hist_of_img, apply_hist_modification_transform
from hist_modif import perform_hist_modification, perform_hist_eq, perform_hist_matching
import os

output_dir = "./output_plots"
os.makedirs(output_dir, exist_ok=True)

# Load images
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

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array, cmap="gray")
plt.title("Αρχική Εικόνα")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.bar(range(len(histogram_normalized)), histogram_normalized.values(), color="blue", alpha=0.7)
plt.title("Κανονικοποιημένο Ιστόγραμμα")
plt.xlabel("Επίπεδο έντασης")
plt.ylabel("Συχνότητα")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "1_original_image_and_histogram.png"))
plt.show()

# Check apply_hist_modification_transform
histogram_transform = apply_hist_modification_transform(img_array, histogram)
print(f"Ιστόγραμμα μετασχηματισμού: {histogram_transform}")

# Check perform_hist_modification
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(img_array, cmap="gray")
plt.title("Αρχική Εικόνα")
plt.colorbar()

methods = ["greedy", "non-greedy", "post-disturbance"]
titles = ["Greedy", "Non-greedy", "Post-disturbance"]
modified_images = []

for i, method in enumerate(methods):
    modified_img = perform_hist_modification(img_array, histogram2_normalized, method)
    modified_images.append(modified_img)
    plt.subplot(1, 4, i+2)
    plt.imshow(modified_img, cmap="gray")
    plt.title(titles[i])
    plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "2a_hist_modification_images.png"))
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.bar(range(len(histogram)), histogram.values(), color="blue", alpha=0.7)
plt.title("Αρχικό Ιστόγραμμα")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

for i, (method, title) in enumerate(zip(methods, titles)):
    hist = calculate_hist_of_img(modified_images[i], return_normalized=True)
    plt.subplot(1, 4, i+2)
    plt.bar(range(len(hist)), hist.values(), color=["green", "red", "purple"][i], alpha=0.7)
    plt.title(f"{title} Ιστόγραμμα")
    plt.xlabel("Στάθμη εισόδου")
    plt.ylabel("Συχνότητα")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "2b_hist_modification_histograms.png"))
plt.show()

# Check perform_hist_eq
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(img_array, cmap="gray")
plt.title("Αρχική Εικόνα")
plt.colorbar()

equalized_images = []
for i, method in enumerate(methods):
    equalized_img = perform_hist_eq(img_array, method)
    equalized_images.append(equalized_img)
    plt.subplot(1, 4, i+2)
    plt.imshow(equalized_img, cmap="gray")
    plt.title(titles[i])
    plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "3a_hist_equalization_images.png"))
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.bar(range(len(histogram)), histogram.values(), color="blue", alpha=0.7)
plt.title("Αρχικό Ιστόγραμμα")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

for i, (method, title) in enumerate(zip(methods, titles)):
    hist = calculate_hist_of_img(equalized_images[i], return_normalized=True)
    plt.subplot(1, 4, i+2)
    plt.bar(range(len(hist)), hist.values(), color=["green", "red", "purple"][i], alpha=0.7)
    plt.title(f"{title} Ιστόγραμμα")
    plt.xlabel("Στάθμη εισόδου")
    plt.ylabel("Συχνότητα")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "3b_hist_equalization_histograms.png"))
plt.show()

# Check perform_hist_matching
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_array2, cmap="gray")
plt.title("Εικόνα Αναφοράς")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.bar(range(len(histogram2)), histogram2.values(), color="orange", alpha=0.7)
plt.title("Ιστόγραμμα Αναφοράς")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "4a_reference_image_and_histogram.png"))
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(img_array, cmap="gray")
plt.title("Αρχική Εικόνα")
plt.colorbar()

matched_images = []
for i, method in enumerate(methods):
    matched_img = perform_hist_matching(img_array, img_array2, method)
    matched_images.append(matched_img)
    plt.subplot(1, 4, i+2)
    plt.imshow(matched_img, cmap="gray")
    plt.title(titles[i])
    plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "4b_hist_matching_images.png"))
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.bar(range(len(histogram)), histogram.values(), color="blue", alpha=0.7)
plt.title("Αρχικό Ιστόγραμμα")
plt.xlabel("Στάθμη εισόδου")
plt.ylabel("Συχνότητα")

for i, (method, title) in enumerate(zip(methods, titles)):
    hist = calculate_hist_of_img(matched_images[i], return_normalized=True)
    plt.subplot(1, 4, i+2)
    plt.bar(range(len(hist)), hist.values(), color=["green", "red", "purple"][i], alpha=0.7)
    plt.title(f"{title} Ιστόγραμμα")
    plt.xlabel("Στάθμη εισόδου")
    plt.ylabel("Συχνότητα")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "4c_hist_matching_histograms.png"))
plt.show()
