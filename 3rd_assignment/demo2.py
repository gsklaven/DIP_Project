import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering
import os


plt.ioff()
output_dir = "./output_plots"
os.makedirs(output_dir, exist_ok=True)

k_list = [2, 3, 4]

data2a = loadmat("dip_hw_3.mat")
d2a = data2a["d2a"]

m2, n2, c2 = d2a.shape
img_graph = image_to_graph(d2a)

for idx, k in enumerate(k_list):
    labels = spectral_clustering(img_graph, k)

    label_img = labels.reshape(m2, n2)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(d2a)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Spectral Clustering (k={k})")
    plt.imshow(label_img, cmap='viridis')
    plt.axis('off')

    plt.savefig(os.path.join(output_dir, f"Data 2a clustering {k}.png"))
    plt.tight_layout()
    plt.show()

data2b = loadmat("dip_hw_3.mat")
d2b = data2b["d2b"]

m3, n3, c3 = d2b.shape
img_graph = image_to_graph(d2b)

for idx, k in enumerate(k_list):
    labels = spectral_clustering(img_graph, k)

    label_img = labels.reshape(m3, n3)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(d2b)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Spectral Clustering (k={k})")
    plt.imshow(label_img, cmap='viridis')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Data 2b clustering_{k}.png"))
    plt.show()
