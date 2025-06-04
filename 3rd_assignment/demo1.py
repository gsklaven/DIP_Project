import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering

# Φόρτωσε δεδομένα
data = loadmat("dip_hw_3.mat")
d2a = data["d2a"]

# Αν η εικόνα είναι [M, N, C]
M, N, C = d2a.shape
img_graph = image_to_graph(d2a)

# Λίστα από αριθμούς clusters
k_list = [2, 3, 4]

for idx, k in enumerate(k_list):
    labels = spectral_clustering(img_graph, k)

    # Μετασχημάτισε τις ετικέτες σε εικόνα
    label_img = labels.reshape(M, N)

    # Εμφάνιση
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(d2a)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Spectral Clustering (k={k})")
    plt.imshow(label_img, cmap='viridis')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
