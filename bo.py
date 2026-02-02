import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
import random
from datetime import datetime
import scipy  # allowed
from group2_bbo import GP  # your GP from group2_bbo.py
from utils import plot_cum_sum
import matplotlib.pyplot as plt

CELLTYPES = ['celltype_1', 'celltype_2', 'celltype_3']
CELL_TO_IDX = {c: i for i, c in enumerate(CELLTYPES)}
IDX_TO_CELL = {i: c for i, c in enumerate(CELLTYPES)}

# ---- objective wrapper (expects list-of-rows) ----
def objective_func(X_rows):
    return np.array(virtual_lab.conduct_experiment(X_rows), dtype=float).reshape(-1)

# -----------------------
# Fixed scaling utilities
# -----------------------
def cont_to_unit(cont5, lo, hi):
    cont5 = np.asarray(cont5, dtype=float)
    return (cont5 - lo) / (hi - lo + 1e-12)

def unit_to_cont(u5, lo, hi):
    u5 = np.asarray(u5, dtype=float)
    u5 = np.clip(u5, 0.0, 1.0)
    return lo + u5 * (hi - lo)

# ---- encoding/decoding categorical (SCALED continuous) ----
def encode_row_scaled(row6, lo, hi):
    # scale continuous to [0,1]
    u = cont_to_unit(row6[:5], lo, hi)
    onehot = np.zeros(3, dtype=float)
    onehot[CELL_TO_IDX[row6[5]]] = 1.0
    return np.concatenate([u, onehot])

def decode_vec_scaled(x8, lo, hi):
    # unit continuous back to original bounds + argmax celltype
    cont = unit_to_cont(x8[:5], lo, hi)
    ct = int(np.argmax(x8[5:8]))
    return [float(cont[0]), float(cont[1]), float(cont[2]), float(cont[3]), float(cont[4]), IDX_TO_CELL[ct]]

def encode_X_scaled(X_rows, lo, hi):
    return np.vstack([encode_row_scaled(r, lo, hi) for r in X_rows])

# ----------------------------
# EI acquisition (maximisation)
# ----------------------------
def normal_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

def normal_cdf(z):
    return 0.5 * (1.0 + scipy.special.erf(z / np.sqrt(2.0)))

def acquisition_ei(mu, sigma, y_best, xi=0.01):
    """
    Expected Improvement for maximisation:
      EI(x) = (mu - y_best - xi) * Phi(Z) + sigma * phi(Z)
      where Z = (mu - y_best - xi) / sigma
    """
    sigma = np.maximum(sigma, 1e-12)
    imp = mu - y_best - xi
    Z = imp / sigma
    return imp * normal_cdf(Z) + sigma * normal_pdf(Z)

# ----------------------------
# Batch diversity in unit space
# ----------------------------
def scaled_dist2_unit(x8_a, x8_b):
    # continuous part only, already in [0,1], so range is 1
    d = 0.0
    for i in range(5):
        diff = x8_a[i] - x8_b[i]
        d += diff * diff
    return d

def select_batch_greedy_diverse(Xcand8, acq, batch, min_dist2=0.01):
    """
    Greedy top-acquisition with a minimum squared distance threshold
    on unit-space continuous vars to avoid near-duplicates in a batch.
    min_dist2=0.01 ~ keep points at least ~0.1 apart in unit distance.
    """
    order = np.argsort(-acq)
    chosen = []
    for idx in order:
        x = Xcand8[int(idx)]
        ok = True
        for xc in chosen:
            if scaled_dist2_unit(x, xc) < min_dist2:
                ok = False
                break
        if ok:
            chosen.append(x)
            if len(chosen) == batch:
                break

    # Top up if needed
    if len(chosen) < batch:
        for idx in order:
            x = Xcand8[int(idx)]
            if not any(np.allclose(x, c) for c in chosen):
                chosen.append(x)
            if len(chosen) == batch:
                break

    return chosen


class BO:
    """
    BO that ONLY relies on your GP having:
      gp.fit(X_numeric, y)
      gp.predict(X_numeric) -> (mean, std)

    Here X_numeric is 8D:
      [u_temp,u_pH,u_f1,u_f2,u_f3, onehot(celltype)]
    where u_* in [0,1].
    """

    def __init__(self, GP_class, iterations=15, batch=5, n_candidates=5000):
        start_time = datetime.timestamp(datetime.now())  # REQUIRED first line

        self.iterations = int(iterations)
        self.batch = int(batch)
        self.n_candidates = int(n_candidates)

        # storage required by brief
        self.Y = []
        self.time = []

        # bounds (original units)
        self.lo = np.array([30.0, 6.0, 0.0, 0.0, 0.0], dtype=float)
        self.hi = np.array([40.0, 8.0, 50.0, 50.0, 50.0], dtype=float)

        # keep inputs too
        self.X_obs_rows = []   # list of 6D rows (with string celltype)
        self.X_obs_num = None  # ndarray (N,8) in unit space
        self.y_obs = None

        # GP instance (leave hypers as you like)
        self.gp = GP_class(kernel='matern52', length_scale=1.0, signal_variance=1.0, noise_level=1e-5)

        # ---- 1) required init: 6 points ----
        X_init = self._init_design_6()
        Y_init = objective_func(X_init)

        self.X_obs_rows = list(X_init)
        self.X_obs_num = encode_X_scaled(self.X_obs_rows, self.lo, self.hi)
        self.y_obs = np.array(Y_init, dtype=float)

        # record timing: one non-zero then pad zeros
        self.Y += list(Y_init)
        self.time += [datetime.timestamp(datetime.now()) - start_time]
        self.time += [0] * (len(Y_init) - 1)
        start_time = datetime.timestamp(datetime.now())

        # ---- 2) BO loop: 15 iterations, batch 5 ----
        for it in range(self.iterations):
            self._it = it  # used by candidate generator schedule

            # fit GP on scaled numeric data
            self.gp.fit(self.X_obs_num, self.y_obs)

            # candidates in unit space + onehot celltype
            Xcand8 = self._make_candidates_numeric(self.n_candidates)

            # predict
            mu, std = self.gp.predict(Xcand8)

            # EI acquisition (maximisation)
            y_best = float(np.max(self.y_obs))
            xi = self._xi_schedule(it)
            acq = acquisition_ei(mu, std, y_best=y_best, xi=xi)

            # batch selection with diversity (unit space)
            chosen8 = select_batch_greedy_diverse(Xcand8, acq, self.batch, min_dist2=0.01)

            # decode to original units for simulator
            X_batch = [decode_vec_scaled(x, self.lo, self.hi) for x in chosen8]

            # evaluate batch
            Y_batch = objective_func(X_batch)

            # append histories
            self.X_obs_rows += list(X_batch)
            self.X_obs_num = np.vstack([self.X_obs_num, np.vstack(chosen8)])
            self.y_obs = np.concatenate([self.y_obs, Y_batch])

            # required outputs + time bookkeeping
            self.Y += list(Y_batch)
            self.time += [datetime.timestamp(datetime.now()) - start_time]
            self.time += [0] * (len(Y_batch) - 1)
            start_time = datetime.timestamp(datetime.now())

    def _xi_schedule(self, it):
        # Stronger exploration early, decay slowly, keep a floor
        return max(0.005, 0.1 * (0.90 ** it))

    def _init_design_6(self):
        # 6 init points with 2 per cell type, continuous uniform in bounds
        X = []
        for ct in CELLTYPES:
            for _ in range(2):
                cont = self.lo + (self.hi - self.lo) * np.random.rand(5)
                X.append([float(cont[0]), float(cont[1]), float(cont[2]), float(cont[3]), float(cont[4]), ct])
        random.shuffle(X)
        return X

    def _make_candidates_numeric(self, n):
        """
        Candidates in 8D numeric:
          - first 5 dims: continuous in [0,1]
          - last 3 dims: one-hot cell type
        Mix:
          - global uniform samples in unit space
          - local Gaussian perturbations around top observed points
        """
        n = int(n)
        n_global = int(0.6 * n)
        n_local = n - n_global

        X8 = []

        # ---- GLOBAL (uniform in unit space) ----
        n_per = max(1, n_global // 3)
        for ct in CELLTYPES:
            onehot = np.zeros(3, dtype=float)
            onehot[CELL_TO_IDX[ct]] = 1.0
            U = np.random.rand(n_per, 5)  # [0,1]
            for i in range(U.shape[0]):
                X8.append(np.concatenate([U[i], onehot]))

        while len(X8) < n_global:
            ct = random.choice(CELLTYPES)
            onehot = np.zeros(3, dtype=float)
            onehot[CELL_TO_IDX[ct]] = 1.0
            u = np.random.rand(5)
            X8.append(np.concatenate([u, onehot]))

        # ---- LOCAL (around top-k observed) ----
        y = self.y_obs
        X = self.X_obs_num
        k = min(6, len(y))
        top_idx = np.argsort(-y)[:k]
        centres = X[top_idx, :5]  # unit cont
        centre_ct = np.argmax(X[top_idx, 5:8], axis=1)

        # sigma decays with iteration => more exploitation later
        sigma = max(0.05, 0.15 * (0.85 ** self._it))

        for _ in range(n_local):
            j = random.randrange(k)
            u = centres[j] + sigma * np.random.randn(5)
            u = np.clip(u, 0.0, 1.0)
            ct = int(centre_ct[j])  # keep celltype of centre
            onehot = np.zeros(3, dtype=float)
            onehot[ct] = 1.0
            X8.append(np.concatenate([u, onehot]))

        return np.vstack(X8[:n])


# ---- run ----
BO_m = BO(GP_class=GP, iterations=15, batch=5, n_candidates=300000)

print("Best titre:", max(BO_m.Y))
i = int(np.argmax(BO_m.Y))

def plot_coursework_cum_sum(bo, ax=None):
    y = np.asarray(bo.Y, dtype=float).ravel()
    dt = np.asarray(bo.time, dtype=float).ravel()
    n = min(len(y), len(dt))
    y, dt = y[:n], dt[:n]

    # Skip the first 16 evaluations (init + batch1 + batch2)
    y = y[16:]
    dt = dt[16:]

    t = np.cumsum(dt)
    starts = np.flatnonzero(dt > 0)
    if starts.size == 0 or starts[0] != 0:
        starts = np.insert(starts, 0, 0)
    ends = np.concatenate([starts[1:], np.array([len(y)], dtype=int)])

    running_total = 0.0
    cum_sum = []
    t_arr = []
    for end in ends:
        best_so_far = float(np.max(y[:end]))
        running_total += best_so_far
        cum_sum.append(running_total)
        t_arr.append(t[end - 1] if end - 1 >= 0 else 0.0)

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(t_arr, cum_sum, linewidth=2)
    ax.scatter(t_arr, cum_sum, s=30, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative score (coursework)")
    ax.set_title("Coursework cumulative score vs time")
    ax.grid(True, alpha=0.3)
    return ax

plot_coursework_cum_sum(BO_m)
plt.show()