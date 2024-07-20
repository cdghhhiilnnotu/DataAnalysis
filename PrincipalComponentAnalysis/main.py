from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from pca import PCA

data = datasets.load_iris()
X,y = data.data, data.target

pca = PCA(2)
pca.fit(X)
X_ = pca.transform(X)

x1 = X_[:,0]
x2 = X_[:,1]

plt.scatter(x1, x2,
            c=y, edgecolors='none', alpha=0.8,
            cmap=plt.cm.get_cmap('viridis', 3))

plt.xlabel("Co 1")
plt.ylabel("Co 2")
plt.colorbar()
plt.show()



