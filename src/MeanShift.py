import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

from scipy.io import loadmat
x = loadmat('/home/sarah/Downloads/DiffSeg-Data/mwu100307/data.mat')
slide = 50
i = x['imgs'][:, :, slide, 0]
X = np.reshape(i, [-1, 1])
print(X.shape)
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
print(bandwidth)

clustering = MeanShift(bandwidth=bandwidth).fit(X)
labels = clustering.labels_
cluster_centers = clustering.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

segmented_image = np.reshape(labels, i.shape)  # Just take size, ignore RGB channels.

plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(i)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.axis('off')

plt.show()