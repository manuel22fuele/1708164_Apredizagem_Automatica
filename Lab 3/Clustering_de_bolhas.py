from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

import numpy as np

from sklearn.datasets import load_digits
data, labels = load_digits(return_X_y=True)

plt.figure(figsize=(12, 12))
n_samples = 1500
random_state = 170
#X, y = make_blobs(n_samples=n_samples, random_state=random_state)
X, y = data, labels = load_digits(return_X_y=True)

# Incorrect number of clusters
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# Plot the centroids as a black/blue O - Calculate from kmeans++
from sklearn.cluster import kmeans_plusplus
kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10)
kmeans.fit(X)
# Plot the centroids as a black/blue O
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=90,
    linewidths=1, color="b", edgecolor="black")
plt.title("Número incorreto de Clusters")

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)
plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
# Plot the centroids as a black/blue O - Calculate from kmeans++
from sklearn.cluster import kmeans_plusplus
kmeans = KMeans(init="k-means++", n_clusters=3, n_init=10)
kmeans.fit(X_aniso)
# Plot the centroids as a black/blue O
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=90,
    linewidths=1, color="b", edgecolor="black")
plt.title("Distribuição anisotrópica de bolhas")

# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
y_pred = KMeans(n_clusters=3,random_state=random_state).fit_predict(X_varied)
plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)

# Plot the centroids as a black/blue O - Calculate from kmeans++
from sklearn.cluster import kmeans_plusplus
kmeans = KMeans(init="k-means++", n_clusters=3, n_init=10)
kmeans.fit(X_varied)
# Plot the centroids as a black/blue O
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=90,
linewidths=1, color="b", edgecolor="black")
plt.title("Diferentes variâncias - Diferentes dispersões")

# Unevenly sized blobs
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
#ou X_filtered, y = make_blobs(n_samples=[500,100,10], random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)
plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
# Plot the centroids as a black/blue O - Calculate from kmeans++
from sklearn.cluster import kmeans_plusplus
kmeans = KMeans(init="k-means++", n_clusters=3, n_init=10)
kmeans.fit(X_filtered)
# Plot the centroids as a black/blue O
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=90,
linewidths=1, color="b", edgecolor="black")
#plt.title("Unevenly Sized Blobs")
plt.title("Clusters com diferentes tamanhos")
plt.show()
