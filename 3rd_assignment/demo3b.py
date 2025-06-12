import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from n_cuts import n_cuts, calculate_n_cut_value
import os


plt.ioff()
output_dir = "./output_plots"
os.makedirs(output_dir, exist_ok=True)

k_list = [2]

data2a = loadmat("dip_hw_3.mat")
d2a = data2a["d2a"]

m2, n2, c2 = d2a.shape
img_graph = image_to_graph(d2a)

for idx, k in enumerate(k_list):
    labels = n_cuts(img_graph, k)
    n_cut_value = calculate_n_cut_value(img_graph, labels)

    label_img = labels.reshape(m2, n2)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(d2a)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title(f"Normalized Cuts (k=2, n_cut={n_cut_value:.2f})")
    plt.imshow(label_img, cmap='viridis')
    plt.axis('off')

    plt.savefig(os.path.join(output_dir, f"Data 2a ncuts_semirecursive_{k}.png"))
    plt.tight_layout()
    plt.show()

data2b = loadmat("dip_hw_3.mat")
d2b = data2b["d2b"]

m3, n3, c3 = d2b.shape
img_graph = image_to_graph(d2b)

for idx, k in enumerate(k_list):
    labels = n_cuts(img_graph, k)
    n_cut_value = calculate_n_cut_value(img_graph, labels)

    label_img = labels.reshape(m3, n3)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(d2b)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title(f"Normalized Cuts (k=2, n_cut={n_cut_value:.2f})")
    plt.imshow(label_img, cmap='viridis')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Data 2b ncuts_semirecursive_{k}.png"))
    plt.show()
