

"""
How to use kmeans clustering algorithm?
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

## Load data
from sklearn.datasets import load_digits
digits = load_digits()

## Build a kemans model
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(digits.data)

print(kmeans.cluster_centers_.shape)

## Visualize the cluster centers
fig = plt.figure(figsize=(8,3))
for i in range(10):
    ax = fig.add_subplot(2,5,1+i)
    ax.imshow(kmeans.cluster_centers_[i].reshape((8,8)),
              cmap=plt.cm.binary)
# Very interesting! Visualizing cluster centers for more than 3D vectors is not possible
# Here we converted the 64-dim vector into 8by8 matrix and plotted is an image
# Interestingly, images correspond to numbers! Wow.

# In order to visualize 64D data on a 2D graph:
from sklearn.manifold import Isomap
X_iso = Isomap(n_neighbors=10).fit_transform(digits.data)

fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].scatter(X_iso[:,0], X_iso[:,1], c=clusters)
ax[0].set_title('Clustered')
ax[1].scatter(X_iso[:,0], X_iso[:,1], c=digits.target)
ax[1].set_title('Actual')

## Evaluate performance
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(digits.target, clusters)
# This could be negative. 1 is perfect score.


