import matplotlib.pyplot as plt
from scipy.io import loadmat
from image_to_graph import image_to_graph
from spectral_clustering import spectral_clustering


data1 = loadmat("dip_hw_3.mat")
d1a = data1["d1a"]

k_list = [2, 3, 4]
for idx, k in enumerate(k_list):
    spectral_clustering(d1a, k)
    print(f"Spectral clustering with k={k} completed.")
    print(spectral_clustering(d1a, k))
