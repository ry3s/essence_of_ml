import numpy as np
from scipy.sparse.linalg import svds


class PCA:
    def __init__(self, n_components, tol=0.0, random_seed=0):
        self.n_components = n_components
        self.tol = tol
        self.random_state = np.random.RandomState(random_seed)

    def fit(self, X):
        v0 = self.random_state.randn(min(X.shape))
        xbar = X.mean(axis=0)
        Y = X - xbar
        S = np.dot(Y.T, Y)
        U, Sigma, VT = svds(S, k=self.n_components, tol=self.tol, v0=v0)

        self.VT_ = VT[::-1, :]

    def transform(self, X):
        return self.VT_.dot(X.T).T
