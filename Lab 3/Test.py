from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import pandas.io.pickle as pd
import pickle as p1

# Lendo o arquivo "Test_predictor_pca" e armazenando na variável "file_test"
file_test = pd.read_pickle('../Lab 3/Datasets/Test_predictor_pca.csv')
# Imprime os dados armazenados em 'file_test'
print(file_test)
X = file_test[:,0:2] # Pegando simplesmente duas linhas
y = file_test[:,1] # Atribui 1 à y, usando simplesmente a primeira coluna


plt.figure(figsize=(12, 12)) # Criando uma nova figura com o tamanho da largura e altura de 12 polegadas respetivamente
n_samples = len(X) # Essa linha de código atribui à variável n_samples o valor do comprimento da lista X, e usei a função len para obter este valor
random_state = 40 # Atribuí a  variável random_state o valor de 40, podendo atribuir qualquer valor...
X, y = make_blobs(n_samples=n_samples, random_state=random_state)


# Incorrect number of clusters
y_pred = KMeans(n_clusters=n_samples, random_state=random_state).fit_predict(X)
plt.subplot(221)
plt.scatter(X[:, 1], X[:, 1], c=y_pred)

# Plot the centroids as a black/blue O - Calculate from kmeans++
kmeans = KMeans(init="k-means++", n_clusters=2, n_init=10)
kmeans.fit(X)
# Plot the centroids as a black/blue O
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=90,
    linewidths=1, color="b", edgecolor="black")
plt.title("Número Teste de Clusters")

plt.show()
