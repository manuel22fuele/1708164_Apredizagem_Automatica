import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as p1
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, linear_model
from pandas.core.common import random_state
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

n_neighbors = 15
random_state = 0
# import some data to play with
data = pd.read_csv("../Lab 1/Datasets/optdigits_tra.txt", sep=",", header=None)
train_data = data[:]
X = train_data.iloc[:,0:64]
y = train_data.iloc[:,64]
h = 2 # step size in the mesh

nca = make_pipeline(StandardScaler(), NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),)
nca.fit(X, y)

# Gravar ficheiro nca
preditor_nca=nca.fit(X, y)
preditor_Pickle = open('../Lab 2/Datasets/optdigits_predictor_nca', 'wb')
p1.dump(preditor_nca, preditor_Pickle)

X = nca.transform(X)

# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue", "red", "blue", "pink", "brown","green","white","purple"])
cmap_bold = ["darkorange", "c", "darkblue","darksalmon", "lightblue","lightyellow","darkgreen","lightgreen","brown","darkgrey"]


clf = neighbors.KNeighborsClassifier(n_neighbors, weights= "uniform")
clf.fit(X, y)
print(clf.score(X, y))

# imports
# Gravar ficheiro clf
preditor_Kneighbor=clf.fit(X, y)
preditor_Pickle = open('../Lab 2/Datasets/optdigits_predictor_clf', 'wb')
p1.dump(preditor_Kneighbor, preditor_Pickle)

# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)
# Plot also the training points
sns.scatterplot(
x=X[:, 0],
y=X[:, 1],
hue=y,
palette=cmap_bold,
alpha=1.0,
legend=True,
edgecolor="black",)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i, weights = '%s')" % (n_neighbors, "uniform"))
plt.xlabel('Valor de X')
plt.ylabel('Valor de Y')

# Plot a predict point
sns.scatterplot(
x=(X[1,0]+X[43,0])/2,
y=(X[1,1]+X[43,1])/2,
marker="X",
s=90,
hue=y,
palette=cmap_bold,
alpha=1.0,
legend=False,
edgecolor="white",)
plt.show()
