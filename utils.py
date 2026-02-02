import numpy as np
import matplotlib.pyplot as plt


def plot_cum_sum(bo, maximize=True, include_init=True, ax=None, show_batch_points=True):

    y = np.asarray(bo.Y, dtype=float).ravel()
    dt = np.asarray(bo.time, dtype=float).ravel()
    n = min(len(y), len(dt)) # in case alg does not run in 60s
    y, dt = y[:n], dt[:n]

    t = np.cumsum(dt)

    # Batch boundaries are where dt > 0.
    starts = np.flatnonzero(dt > 0)
    if starts.size == 0 or starts[0] != 0:
        starts = np.insert(starts, 0, 0)

    if not include_init and starts.size > 1:
        starts = starts[1:]

    ends = np.concatenate([starts[1:], np.array([n], dtype=int)])

    running_total = 0.0
    cum_sum = []
    t_arr = []

    for end in ends:
        y_prefix = y[:end]
        b = float(np.max(y_prefix)) if maximize else float(np.min(y_prefix))
        running_total += b
        cum_sum.append(running_total)
        t_arr.append(t[end - 1] if end - 1 >= 0 else 0.0)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(t_arr, cum_sum, linewidth=2)
    if show_batch_points:
        ax.scatter(t_arr, cum_sum, s=30, alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative sum of current best (per batch)")
    ax.set_title("Cumulative objective over time (sum of global best after each batch)")
    ax.grid(True, alpha=0.3)
    return ax