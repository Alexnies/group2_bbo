import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, solve_triangular


# Custom GP with multiple kernel options
class GP:    
    def __init__(self, kernel='matern52', length_scale=1.0, signal_variance=1.0, noise_level=1e-5):
        self.kernel = kernel.lower()
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_level = noise_level
        
        self.X_train = None
        self.y_train = None
        self.L = None  # Cholesky factor
        self.alpha = None  # K^-1 @ y
        
        self._X_min = None
        self._X_max = None
        self.y_mean = 0.0
        self.y_std = 1.0
    
    def _normalise_X(self, X):
        return (X - self._X_min) / (self._X_max - self._X_min + 1e-10)
    
    def _rbf_kernel(self, X1, X2):
        dist = cdist(X1, X2, metric='euclidean')
        return np.exp(-0.5 * (dist / self.length_scale) ** 2)
    
    def _matern32_kernel(self, X1, X2):
        dist = cdist(X1, X2, metric='euclidean')
        scaled_dist = np.sqrt(3.0) * dist / self.length_scale
        return (1.0 + scaled_dist) * np.exp(-scaled_dist)
    
    def _matern52_kernel(self, X1, X2):
        dist = cdist(X1, X2, metric='euclidean')
        scaled_dist = np.sqrt(5.0) * dist / self.length_scale
        return (1.0 + scaled_dist + (scaled_dist ** 2) / 3.0) * np.exp(-scaled_dist)
    
    def _compute_kernel(self, X1, X2):
        if self.kernel == 'rbf':
            K = self._rbf_kernel(X1, X2)
        elif self.kernel == 'matern32':
            K = self._matern32_kernel(X1, X2)
        elif self.kernel == 'matern52':
            K = self._matern52_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}. Use 'rbf', 'matern32', or 'matern52'.")
        
        return self.signal_variance * K
    
    def negative_loglikelihood(self, X, y):
        log_det_K = 2 * np.sum(np.log(np.diag(self.L)))
        nll = np.dot(y.T, self.alpha) + log_det_K
        return nll
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        
        self._X_min = np.min(X, axis=0)
        self._X_max = np.max(X, axis=0)
        self.X_train = self._normalise_X(X)
        
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-10
        self.y_train = (y - self.y_mean) / self.y_std
        
        K = self._compute_kernel(self.X_train, self.X_train)
        K[np.diag_indices_from(K)] += self.noise_level
        K = (K + K.T) * 0.5

        try:
            self.L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            K[np.diag_indices_from(K)] += 1e-4
            self.L = cholesky(K, lower=True)
        
        M = solve_triangular(self.L, self.y_train, lower=True)
        self.alpha = solve_triangular(self.L.T, M, lower=False)
    
    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        X_test_norm = self._normalise_X(X_test)
        
        K_ = self._compute_kernel(self.X_train, X_test_norm)
        
        y_pred_norm = K_.T @ self.alpha
        mean = y_pred_norm * self.y_std + self.y_mean
        
        v = solve_triangular(self.L, K_, lower=True)
        var_norm = 1.0 - np.sum(v ** 2, axis=0)
        var_norm = np.maximum(var_norm, 0)
        std = np.sqrt(var_norm) * self.y_std
        
        return mean, std
    
    