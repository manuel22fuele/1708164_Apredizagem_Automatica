from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
import pickle as p1

file_train = p1.load(open('../Lab 3/Datasets/Train_predictor_pca.csv', 'rb'))
print(file_train)
X = file_train[:,0:2]
y = 1


plt.figure(figsize=(12, 12))
n_samples = len(X)
random_state = 40
X, y = make_blobs(n_samples=n_samples, random_state=random_state)


# Incorrect number of clusters
y_pred = KMeans(n_clusters=n_samples, random_state=random_state).fit_predict(X)
plt.subplot(221)
plt.scatter(X[:, 1], X[:, 1], c=y_pred)
# Plot the centroids as a black/blue O - Calculate from kmeans++
from sklearn.cluster import kmeans_plusplus
kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10)
kmeans.fit(X)
# Plot the centroids as a black/blue O
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=90,
    linewidths=1, color="b", edgecolor="black")
plt.title("Número Treino de Clusters")

plt.show()
