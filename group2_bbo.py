import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
from datetime import datetime
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, solve_triangular

def objective_func(X: list): 
    return(np.array(virtual_lab.conduct_experiment(X)))

ENCODE_KEY = {'celltype_1': 0, 'celltype_2': 1, 'celltype_3': 2}
DECODE_KEY = {0: 'celltype_1', 1: 'celltype_2', 2: 'celltype_3'}

def encode(X):
    encoded_celltype = []
    for row in X:
        temp, pH, f1, f2, f3, cell_type = row
        celltype_id = ENCODE_KEY.get(str(cell_type), 0)
        encoded_celltype.append([float(temp), float(pH), float(f1), float(f2), float(f3), float(celltype_id)])
    return np.array(encoded_celltype, dtype=float)


class GP:    
    def __init__(self, kernel='matern52', length_scale=1.0, signal_variance=1.0, 
                 noise_level=1e-5, theta=0.25, use_mixed_kernel=False):
        self.kernel = kernel.lower()
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_level = noise_level
        self.theta = theta
        self.use_mixed_kernel = use_mixed_kernel
        
        self.X_train = None
        self.X_train_cat = None
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
    
    def _continuous_kernel(self, X1, X2):
        if self.kernel == 'rbf':
            K = self._rbf_kernel(X1, X2)
        elif self.kernel == 'matern32':
            K = self._matern32_kernel(X1, X2)
        elif self.kernel == 'matern52':
            K = self._matern52_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}. Use 'rbf', 'matern32', or 'matern52'.")
        
        return self.signal_variance * K
    
    def _categorical_kernel(self, cat1, cat2):
        cat1 = np.asarray(cat1).reshape(-1, 1)
        cat2 = np.asarray(cat2).reshape(-1, 1)
        same_category = (cat1 == cat2.T).astype(float)
        return same_category + self.theta * (1 - same_category)
    
    def _compute_kernel(self, X1, X2, cat1=None, cat2=None):
        K_cont = self._continuous_kernel(X1, X2)
        
        if self.use_mixed_kernel and cat1 is not None and cat2 is not None:
            K_cat = self._categorical_kernel(cat1, cat2)
            return K_cont * K_cat
        
        return K_cont
    
    def negative_loglikelihood(self, X, y):
        log_det_K = 2 * np.sum(np.log(np.diag(self.L)))
        nll = np.dot(y.T, self.alpha) + log_det_K
        return nll
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        
        if self.use_mixed_kernel:
            X_cont = X[:, :5]
            self.X_train_cat = X[:, 5].astype(int)
        else:
            X_cont = X
            self.X_train_cat = None
        
        self._X_min = np.min(X_cont, axis=0)
        self._X_max = np.max(X_cont, axis=0)
        self.X_train = self._normalise_X(X_cont)
        
        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-10
        self.y_train = (y - self.y_mean) / self.y_std
        
        K = self._compute_kernel(self.X_train, self.X_train, 
                                 self.X_train_cat, self.X_train_cat)
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
        
        if self.use_mixed_kernel:
            X_test_cont = X_test[:, :5]
            X_test_cat = X_test[:, 5].astype(int)
        else:
            X_test_cont = X_test
            X_test_cat = None
        
        X_test_norm = self._normalise_X(X_test_cont)
        
        K_ = self._compute_kernel(self.X_train, X_test_norm,
                                  self.X_train_cat, X_test_cat)
        
        y_pred_norm = K_.T @ self.alpha
        mean = y_pred_norm * self.y_std + self.y_mean
        
        v = solve_triangular(self.L, K_, lower=True)
        var_norm = self.signal_variance - np.sum(v ** 2, axis=0)
        var_norm = np.maximum(var_norm, 0)
        std = np.sqrt(var_norm) * self.y_std
        
        return mean, std
    


    
    