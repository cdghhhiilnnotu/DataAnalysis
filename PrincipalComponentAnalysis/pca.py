import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.cooponents = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X  -= self.mean

        cov = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[0:self.n_components]


    def transform(self, X):
        X -= self.mean
        return np.dot(X, self.components.T)
